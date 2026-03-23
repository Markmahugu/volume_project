"""Height-based filters and local ground modeling for stockpile isolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from src.preprocess import estimate_mean_point_spacing


@dataclass(slots=True)
class GroundModel:
    origin_xy: np.ndarray
    cell_size: float
    z_grid: np.ndarray
    valid_mask: np.ndarray
    global_z_min: float


def build_ground_model(
    ground_cloud: o3d.geometry.PointCloud,
    cell_size: float | None = None,
) -> GroundModel:
    """Build a local ground-height lookup grid from ground points."""
    if ground_cloud.is_empty():
        raise ValueError("Cannot build a ground model from an empty point cloud.")

    points = np.asarray(ground_cloud.points, dtype=float)
    if cell_size is None:
        cell_size = max(estimate_mean_point_spacing(ground_cloud) * 3.0, 0.05)

    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    dims = np.maximum(np.ceil((max_xy - min_xy) / cell_size).astype(int) + 1, 1)

    z_grid = np.full((dims[1], dims[0]), np.nan, dtype=float)
    for x, y, z in points:
        ix = min(int((x - min_xy[0]) / cell_size), dims[0] - 1)
        iy = min(int((y - min_xy[1]) / cell_size), dims[1] - 1)
        current = z_grid[iy, ix]
        z_grid[iy, ix] = z if np.isnan(current) else min(current, z)

    valid_mask = np.isfinite(z_grid)
    return GroundModel(
        origin_xy=min_xy,
        cell_size=float(cell_size),
        z_grid=z_grid,
        valid_mask=valid_mask,
        global_z_min=float(points[:, 2].min()),
    )


def estimate_ground_heights(
    points_xy: np.ndarray,
    ground_model: GroundModel,
    search_radius: int = 2,
) -> np.ndarray:
    """Estimate local ground height for arbitrary XY queries using nearby grid cells."""
    if points_xy.size == 0:
        return np.empty((0,), dtype=float)

    heights = np.empty(points_xy.shape[0], dtype=float)
    grid_h, grid_w = ground_model.z_grid.shape

    for index, (x, y) in enumerate(points_xy):
        ix = int((x - ground_model.origin_xy[0]) / ground_model.cell_size)
        iy = int((y - ground_model.origin_xy[1]) / ground_model.cell_size)

        best_values = []
        best_weights = []
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                gx = ix + dx
                gy = iy + dy
                if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
                    continue
                if not ground_model.valid_mask[gy, gx]:
                    continue
                cell_x = ground_model.origin_xy[0] + (gx + 0.5) * ground_model.cell_size
                cell_y = ground_model.origin_xy[1] + (gy + 0.5) * ground_model.cell_size
                distance = np.hypot(x - cell_x, y - cell_y)
                weight = 1.0 / max(distance, ground_model.cell_size * 0.25)
                best_values.append(ground_model.z_grid[gy, gx])
                best_weights.append(weight)

        if best_values:
            heights[index] = float(np.average(best_values, weights=best_weights))
        else:
            heights[index] = ground_model.global_z_min

    return heights


def compute_heights_above_ground(
    pcd: o3d.geometry.PointCloud,
    ground_model: GroundModel,
) -> np.ndarray:
    """Compute height above the local ground surface for each point."""
    if pcd.is_empty():
        raise ValueError("Cannot compute heights for an empty point cloud.")

    points = np.asarray(pcd.points, dtype=float)
    ground_heights = estimate_ground_heights(points[:, :2], ground_model)
    return points[:, 2] - ground_heights


def height_filter(
    pcd: o3d.geometry.PointCloud,
    ground_model: GroundModel,
    threshold: float = 0.0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    """Keep points at or above the local ground surface plus an optional margin."""
    if pcd.is_empty():
        raise ValueError("Cannot apply a height filter to an empty point cloud.")
    if threshold < 0:
        raise ValueError("height threshold must be non-negative.")

    heights = compute_heights_above_ground(pcd, ground_model)
    keep_indices = np.where(heights >= threshold)[0]
    if keep_indices.size == 0:
        raise RuntimeError("Height filtering removed all points. Lower the height threshold.")

    filtered = pcd.select_by_index(keep_indices.tolist())
    filtered_heights = heights[keep_indices]
    ground_rmse = float(np.sqrt(np.mean(np.square(np.clip(heights, None, 0.0)))))
    return filtered, filtered_heights, ground_rmse
