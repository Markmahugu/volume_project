"""Region-of-interest utilities for user-guided volume selection."""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from matplotlib.path import Path


MIN_Z_SPAN = 0.25
MAX_ROI_VOLUME_M3 = 50.0


def _order_polygon_vertices(points_xy: np.ndarray) -> np.ndarray:
    centroid = points_xy.mean(axis=0)
    angles = np.arctan2(points_xy[:, 1] - centroid[1], points_xy[:, 0] - centroid[0])
    return points_xy[np.argsort(angles)]


def _expand_polygon(points_xy: np.ndarray, padding_xy: float) -> np.ndarray:
    ordered = _order_polygon_vertices(points_xy)
    centroid = ordered.mean(axis=0)
    expanded = []
    for vertex in ordered:
        direction = vertex - centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            expanded.append(vertex)
        else:
            expanded.append(vertex + (direction / norm) * padding_xy)
    return np.asarray(expanded, dtype=float)


def _bbox_volume(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.prod(extent))


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(x), float(y)) for x, y in points_xy})
    if len(ordered) < 3:
        return np.asarray(ordered, dtype=float)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - origin[0]) * (b[1] - origin[1])) - ((a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _filter_by_xy_polygon(points: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if polygon_xy.shape[0] < 3:
        return np.arange(points.shape[0])
    inside_xy = Path(polygon_xy).contains_points(points[:, :2], radius=1e-9)
    return np.where(inside_xy)[0]


def _density_tighten_roi(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 16:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    xy = points[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    cell_size = float(max(min(span) / 30.0, 0.08))

    grid_indices = np.floor((xy - min_xy) / cell_size).astype(int)
    unique_cells, counts = np.unique(grid_indices, axis=0, return_counts=True)
    if counts.size == 0:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    min_count = max(int(np.percentile(counts, 35)), 2)
    dense_cells = {tuple(cell.tolist()) for cell, count in zip(unique_cells, counts) if int(count) >= min_count}
    if not dense_cells:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    expanded_cells = set(dense_cells)
    for cell_x, cell_y in list(dense_cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                expanded_cells.add((cell_x + dx, cell_y + dy))

    keep_mask = np.array([tuple(cell.tolist()) in expanded_cells for cell in grid_indices], dtype=bool)
    tightened_points = points[keep_mask]
    polygon_xy = _convex_hull_xy(tightened_points[:, :2])
    polygon_indices = _filter_by_xy_polygon(points, polygon_xy)
    return points[polygon_indices], polygon_xy


def _validate_roi_volume(points: np.ndarray) -> str | None:
    roi_volume = _bbox_volume(points)
    if roi_volume > MAX_ROI_VOLUME_M3:
        return f"ROI is broad ({roi_volume:.2f} m^3 before ground removal); continuing with a warning because spread-out objects can still be valid."
    return None


def _coerce_cuboid(selection_cuboid: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float, bool, float | None]:
    center = np.asarray(selection_cuboid["center"], dtype=float)
    dimensions = np.asarray(selection_cuboid["dimensions"], dtype=float)
    yaw = float(selection_cuboid.get("yaw", 0.0))
    snap_to_ground = bool(selection_cuboid.get("snap_to_ground", False))
    ground_z = selection_cuboid.get("ground_z")
    ground_z = None if ground_z is None else float(ground_z)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("Cuboid selection must define 3D center and dimensions.")
    if np.any(dimensions <= 0):
        raise ValueError("Cuboid dimensions must be positive.")
    return center, dimensions, yaw, snap_to_ground, ground_z


def transform_points_to_cuboid_local(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    translated = points - center
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = translated[:, 0] * cos_yaw + translated[:, 1] * sin_yaw
    local_y = -translated[:, 0] * sin_yaw + translated[:, 1] * cos_yaw
    local_z = translated[:, 2]
    return np.column_stack([local_x, local_y, local_z])


def compute_cuboid_mask(points: np.ndarray, center: np.ndarray, dimensions: np.ndarray, yaw: float) -> np.ndarray:
    local = transform_points_to_cuboid_local(points, center, yaw)
    half = dimensions / 2.0
    return (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (np.abs(local[:, 2]) <= half[2])
    )


def estimate_ground_z_for_cuboid(
    pcd: o3d.geometry.PointCloud,
    center: np.ndarray,
    dimensions: np.ndarray,
    yaw: float,
    margin: float = 0.2,
) -> float:
    points = np.asarray(pcd.points, dtype=float)
    local = transform_points_to_cuboid_local(points, center, yaw)
    half = dimensions / 2.0
    xy_mask = (
        (np.abs(local[:, 0]) <= half[0] + margin)
        & (np.abs(local[:, 1]) <= half[1] + margin)
    )
    candidates = points[xy_mask, 2]
    if candidates.size == 0:
        return float(points[:, 2].min())
    return float(np.percentile(candidates, 5))


def compute_cuboid_stats(
    pcd: o3d.geometry.PointCloud,
    selection_cuboid: dict[str, object],
) -> dict[str, object]:
    if pcd.is_empty():
        raise ValueError("Cannot analyze an empty point cloud.")

    center, dimensions, yaw, snap_to_ground, ground_z = _coerce_cuboid(selection_cuboid)
    if snap_to_ground:
        inferred_ground_z = estimate_ground_z_for_cuboid(pcd, center, dimensions, yaw)
        if ground_z is None or abs(ground_z - inferred_ground_z) > 1e-6:
            center = center.copy()
            center[2] = inferred_ground_z + (dimensions[2] / 2.0)
            ground_z = inferred_ground_z
    points = np.asarray(pcd.points, dtype=float)
    mask = compute_cuboid_mask(points, center, dimensions, yaw)
    selected_count = int(np.count_nonzero(mask))
    total_count = int(points.shape[0])
    return {
        "center": center.tolist(),
        "dimensions": dimensions.tolist(),
        "yaw": yaw,
        "ground_z": ground_z,
        "selected_points": selected_count,
        "excluded_points": total_count - selected_count,
        "bounding_box_volume_m3": float(np.prod(dimensions)),
    }


def filter_by_cuboid(
    pcd: o3d.geometry.PointCloud,
    selection_cuboid: dict[str, object],
) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")

    center, dimensions, yaw, snap_to_ground, ground_z = _coerce_cuboid(selection_cuboid)
    stats = compute_cuboid_stats(pcd, selection_cuboid)
    center = np.asarray(stats["center"], dtype=float)
    dimensions = np.asarray(stats["dimensions"], dtype=float)
    yaw = float(stats["yaw"])
    points = np.asarray(pcd.points, dtype=float)
    mask = compute_cuboid_mask(points, center, dimensions, yaw)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise RuntimeError("The cuboid does not contain any points.")
    return pcd.select_by_index(indices.tolist()), stats


def compute_polygon_from_picks(
    picked_points: np.ndarray,
    padding_xy: float = 0.0,
) -> np.ndarray:
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target object region.")

    polygon = np.asarray(picked_points[:, :2], dtype=float)
    if padding_xy > 0:
        polygon = _expand_polygon(polygon, padding_xy)
    else:
        polygon = _order_polygon_vertices(polygon)
    return polygon


def filter_by_polygon(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target object region.")

    polygon = compute_polygon_from_picks(picked_points, padding_xy=padding_xy)
    z_values = np.asarray(picked_points[:, 2], dtype=float)
    z_span = max(float(z_values.max() - z_values.min()), MIN_Z_SPAN)
    z_min = float(z_values.min() - max(padding_z, z_span * 2.0))
    z_max = float(z_values.max() + max(padding_z, z_span))

    all_points = np.asarray(pcd.points)
    inside_xy = Path(polygon).contains_points(all_points[:, :2], radius=1e-9)
    inside_z = (all_points[:, 2] >= z_min) & (all_points[:, 2] <= z_max)
    indices = np.where(inside_xy & inside_z)[0]
    if indices.size == 0:
        raise RuntimeError("The selected polygon ROI does not contain any points.")

    roi_points = all_points[indices]
    roi_warning = _validate_roi_volume(roi_points)
    tightened_points, tightened_polygon = _density_tighten_roi(roi_points)
    tightened_warning = _validate_roi_volume(tightened_points)
    if roi_warning:
        print(roi_warning)
    if tightened_warning:
        print(tightened_warning)

    tightened_cloud = o3d.geometry.PointCloud()
    tightened_cloud.points = o3d.utility.Vector3dVector(tightened_points)
    return tightened_cloud, tightened_polygon


def filter_by_bounds(
    pcd: o3d.geometry.PointCloud,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
    full_height: bool = True,
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")

    bounds_min = np.asarray(bounds_min, dtype=float).copy()
    bounds_max = np.asarray(bounds_max, dtype=float).copy()
    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("Selection bounds must each contain exactly 3 coordinates.")
    if np.any(bounds_max < bounds_min):
        raise ValueError("Selection bounds are invalid.")

    all_points = np.asarray(pcd.points, dtype=float)
    bounds_min[0] -= padding_xy
    bounds_min[1] -= padding_xy
    bounds_max[0] += padding_xy
    bounds_max[1] += padding_xy
    if full_height:
        bounds_min[2] = float(all_points[:, 2].min() - padding_z)
        bounds_max[2] = float(all_points[:, 2].max() + padding_z)
    else:
        bounds_min[2] -= padding_z
        bounds_max[2] += padding_z

    bbox = o3d.geometry.AxisAlignedBoundingBox(bounds_min, bounds_max)
    cropped = pcd.crop(bbox)
    if cropped.is_empty():
        raise RuntimeError("The selected bounding box does not contain any points.")

    cropped_points = np.asarray(cropped.points, dtype=float)
    roi_warning = _validate_roi_volume(cropped_points)
    tightened_points, _ = _density_tighten_roi(cropped_points)
    tightened_warning = _validate_roi_volume(tightened_points)
    if roi_warning:
        print(roi_warning)
    if tightened_warning:
        print(tightened_warning)

    tightened_cloud = o3d.geometry.PointCloud()
    tightened_cloud.points = o3d.utility.Vector3dVector(tightened_points)
    return tightened_cloud


def compute_seed_center(picked_points: np.ndarray) -> np.ndarray:
    if picked_points.size == 0:
        raise ValueError("No picked points were provided.")
    return picked_points.mean(axis=0)
