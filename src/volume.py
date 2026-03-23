"""Volume estimation methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from src.preprocess import estimate_mean_point_spacing


@dataclass(slots=True)
class WeightedVoxelMetrics:
    voxel_size: float
    occupied_voxels: int
    total_voxels: int
    empty_ratio: float
    max_expected_density: float


@dataclass(slots=True)
class HeightMapMetrics:
    cell_size: float
    empty_ratio_before_fill: float
    filled_cell_ratio: float


@dataclass(slots=True)
class ValidationSummary:
    adaptive_voxel_size: float
    voxel_volume_m3: float
    weighted_voxel_volume_m3: float
    height_map_volume_m3: float
    mesh_volume_m3: float
    final_volume_m3: float
    method_used: str
    warnings: list[str]
    voxel_metrics: WeightedVoxelMetrics
    height_map_metrics: HeightMapMetrics


def compute_bounding_box_volume(
    pcd: o3d.geometry.PointCloud,
) -> tuple[float, o3d.geometry.AxisAlignedBoundingBox]:
    """Estimate volume using an axis-aligned bounding box for comparison only."""
    if pcd.is_empty():
        raise ValueError("Cannot compute volume for an empty point cloud.")

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=float)
    volume = float(np.prod(extent))
    return volume, bbox


def estimate_adaptive_voxel_size(
    pcd: o3d.geometry.PointCloud,
    fallback: float = 0.02,
) -> float:
    """Choose voxel size from mean point spacing, but clamp it to avoid coarse voxels."""
    spacing = estimate_mean_point_spacing(pcd)
    return float(min(max(fallback * 0.5, spacing * 2.0), 0.05))


def compute_voxel_volume(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[float, o3d.geometry.VoxelGrid]:
    """Estimate volume from occupied voxels."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")
    if pcd.is_empty():
        raise ValueError("Cannot compute volume for an empty point cloud.")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    occupied_voxels = len(voxel_grid.get_voxels())
    volume = occupied_voxels * (voxel_size ** 3)
    return float(volume), voxel_grid


def voxel_grid_to_point_cloud(
    voxel_grid: o3d.geometry.VoxelGrid,
    color: tuple[float, float, float] = (0.12, 0.86, 1.0),
) -> o3d.geometry.PointCloud:
    """Convert occupied voxel centers into a point cloud for visualization."""
    voxels = voxel_grid.get_voxels()
    if not voxels:
        raise ValueError("Voxel grid does not contain any occupied voxels.")

    voxel_size = float(voxel_grid.voxel_size)
    origin = np.asarray(voxel_grid.origin, dtype=np.float64)
    centers = np.asarray(
        [origin + (np.asarray(voxel.grid_index, dtype=np.float64) + 0.5) * voxel_size for voxel in voxels],
        dtype=np.float64,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(centers)
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (centers.shape[0], 1))
    )
    return point_cloud


def create_height_normalized_cloud(
    pcd: o3d.geometry.PointCloud,
    heights: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Build a cloud whose Z values represent height above local ground."""
    if pcd.is_empty():
        raise ValueError("Cannot normalize an empty point cloud.")
    if len(heights) != len(pcd.points):
        raise ValueError("Height count must match point count.")

    points = np.asarray(pcd.points, dtype=np.float64).copy()
    points[:, 2] = np.maximum(heights, 0.0)
    normalized = o3d.geometry.PointCloud()
    normalized.points = o3d.utility.Vector3dVector(points)
    if pcd.has_colors():
        normalized.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors, dtype=np.float64))
    return normalized


def compute_weighted_voxel_volume(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[float, WeightedVoxelMetrics, o3d.geometry.VoxelGrid]:
    """Estimate volume with partial occupancy weighting per voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")
    if pcd.is_empty():
        raise ValueError("Cannot compute a weighted voxel volume for an empty point cloud.")

    points = np.asarray(pcd.points, dtype=np.float64)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    grid_indices = np.floor((points - min_bound) / voxel_size).astype(int)

    unique_indices, counts = np.unique(grid_indices, axis=0, return_counts=True)
    expected_density = float(max(np.percentile(counts, 90), 1.0))
    fill_ratios = np.minimum(1.0, counts / expected_density)
    volume = float(np.sum(fill_ratios) * (voxel_size ** 3))

    dims = np.maximum(np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 1, 1)
    total_voxels = int(np.prod(dims))
    occupied_voxels = int(unique_indices.shape[0])
    empty_ratio = float(max(total_voxels - occupied_voxels, 0) / max(total_voxels, 1))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    metrics = WeightedVoxelMetrics(
        voxel_size=float(voxel_size),
        occupied_voxels=occupied_voxels,
        total_voxels=total_voxels,
        empty_ratio=empty_ratio,
        max_expected_density=expected_density,
    )
    return volume, metrics, voxel_grid


def compute_height_map_volume(
    pcd: o3d.geometry.PointCloud,
    cell_size: float,
) -> tuple[float, HeightMapMetrics]:
    """Estimate volume by integrating a 2.5D height map with gap filling."""
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than zero.")
    if pcd.is_empty():
        raise ValueError("Cannot compute a height-map volume for an empty point cloud.")

    points = np.asarray(pcd.points, dtype=np.float64)
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    dims = np.maximum(np.ceil((max_xy - min_xy) / cell_size).astype(int) + 1, 1)

    height_grid = np.full((dims[1], dims[0]), np.nan, dtype=float)
    for x, y, z in points:
        ix = min(int((x - min_xy[0]) / cell_size), dims[0] - 1)
        iy = min(int((y - min_xy[1]) / cell_size), dims[1] - 1)
        current = height_grid[iy, ix]
        height_grid[iy, ix] = z if np.isnan(current) else max(current, z)

    empty_ratio_before_fill = float(np.count_nonzero(~np.isfinite(height_grid)) / height_grid.size)

    filled = height_grid.copy()
    for _ in range(3):
        missing = ~np.isfinite(filled)
        if not np.any(missing):
            break
        updated = filled.copy()
        ys, xs = np.where(missing)
        for iy, ix in zip(ys, xs):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    gx = ix + dx
                    gy = iy + dy
                    if gx < 0 or gx >= dims[0] or gy < 0 or gy >= dims[1]:
                        continue
                    value = filled[gy, gx]
                    if np.isfinite(value):
                        neighbors.append(value)
            if neighbors:
                updated[iy, ix] = float(np.mean(neighbors))
        filled = updated

    filled = np.nan_to_num(filled, nan=0.0)
    volume = float(np.sum(filled) * (cell_size ** 2))
    filled_cell_ratio = float(np.count_nonzero(filled > 0) / max(filled.size, 1))
    metrics = HeightMapMetrics(
        cell_size=float(cell_size),
        empty_ratio_before_fill=empty_ratio_before_fill,
        filled_cell_ratio=filled_cell_ratio,
    )
    return volume, metrics


def _triangle_mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError("Mesh is empty.")

    total = 0.0
    for i0, i1, i2 in triangles:
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        total += np.dot(p0, np.cross(p1, p2)) / 6.0
    return float(abs(total))


def compute_alpha_shape_volume(
    pcd: o3d.geometry.PointCloud,
    alpha: float,
) -> float:
    """Estimate volume from an alpha-shape mesh."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")
    if pcd.is_empty():
        raise ValueError("Cannot compute a mesh volume for an empty point cloud.")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return _triangle_mesh_volume(mesh)


def compute_validation_volumes(
    pcd: o3d.geometry.PointCloud,
    fallback_voxel_size: float = 0.02,
) -> ValidationSummary:
    """Run multiple volume methods and choose the most stable final estimate."""
    adaptive_voxel_size = estimate_adaptive_voxel_size(pcd, fallback=fallback_voxel_size)
    binary_voxel_volume, _ = compute_voxel_volume(pcd, adaptive_voxel_size)
    weighted_voxel_volume, voxel_metrics, _ = compute_weighted_voxel_volume(pcd, adaptive_voxel_size)
    height_map_volume, height_map_metrics = compute_height_map_volume(pcd, adaptive_voxel_size)

    mesh_candidates = []
    spacing = estimate_mean_point_spacing(pcd)
    for scale in (2.0, 3.0, 4.0):
        alpha = max(spacing * scale, adaptive_voxel_size * 1.5)
        try:
            mesh_candidates.append(compute_alpha_shape_volume(pcd, alpha))
        except Exception:
            continue
    mesh_volume = float(np.median(mesh_candidates)) if mesh_candidates else height_map_volume

    warnings: list[str] = []
    volumes = [binary_voxel_volume, weighted_voxel_volume, height_map_volume, mesh_volume]
    positive_volumes = [value for value in volumes if value > 0]
    spread_ratio = max(positive_volumes) / max(min(positive_volumes), 1e-6) if positive_volumes else float('inf')

    if voxel_metrics.empty_ratio > 0.9:
        warnings.append('Voxel grid is sparse; voxel-based estimates are unreliable.')

    priority = [
        ('mesh', mesh_volume),
        ('weighted_voxel', weighted_voxel_volume),
        ('voxel', binary_voxel_volume),
        ('height_map', height_map_volume),
    ]
    method_used = 'mesh'
    final_volume = float(mesh_volume)

    if spread_ratio > 3.0:
        warnings.append('Volume methods diverged strongly; falling back to mesh volume.')
    else:
        for method_name, value in priority:
            if value > 0:
                method_used = method_name
                final_volume = float(value)
                break

    if mesh_volume < (binary_voxel_volume / 3.0):
        warnings.append('Voxel estimate is much larger than mesh estimate; overestimation likely.')
    if final_volume > 10.0:
        warnings.append('Final volume is larger than the expected sand pile range.')

    return ValidationSummary(
        adaptive_voxel_size=adaptive_voxel_size,
        voxel_volume_m3=float(binary_voxel_volume),
        weighted_voxel_volume_m3=float(weighted_voxel_volume),
        height_map_volume_m3=float(height_map_volume),
        mesh_volume_m3=float(mesh_volume),
        final_volume_m3=final_volume,
        method_used=method_used,
        warnings=warnings,
        voxel_metrics=voxel_metrics,
        height_map_metrics=height_map_metrics,
    )
