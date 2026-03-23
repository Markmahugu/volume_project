"""Region-of-interest utilities for user-guided sand pile selection."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from matplotlib.path import Path


MIN_Z_SPAN = 0.25


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


def compute_polygon_from_picks(
    picked_points: np.ndarray,
    padding_xy: float = 0.0,
) -> np.ndarray:
    """Build a padded 2D polygon from user-picked points in the XY plane."""
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target sand region.")

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
    """Keep only points inside the picked XY polygon and a padded Z slab."""
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target sand region.")

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

    return pcd.select_by_index(indices.tolist()), polygon


def compute_seed_center(picked_points: np.ndarray) -> np.ndarray:
    """Compute the centroid of the user-picked points."""
    if picked_points.size == 0:
        raise ValueError("No picked points were provided.")

    return picked_points.mean(axis=0)
