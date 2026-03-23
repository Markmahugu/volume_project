"""Utilities for loading point cloud files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(slots=True)
class PointCloudValidationSummary:
    total_points: int
    removed_points: int
    removed_ratio: float


def _sanitize_point_cloud(point_cloud: o3d.geometry.PointCloud) -> PointCloudValidationSummary:
    points = np.asarray(point_cloud.points, dtype=np.float64)
    total_points = int(points.shape[0])
    finite_mask = np.all(np.isfinite(points), axis=1)
    bounded_mask = np.all(np.abs(points) < 1000.0, axis=1)
    valid_mask = finite_mask & bounded_mask

    sanitized_points = points[valid_mask]
    removed_points = total_points - int(sanitized_points.shape[0])
    removed_ratio = float(removed_points / max(total_points, 1))
    summary = PointCloudValidationSummary(
        total_points=total_points,
        removed_points=removed_points,
        removed_ratio=removed_ratio,
    )

    if removed_points > 0:
        point_cloud.points = o3d.utility.Vector3dVector(sanitized_points)
        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors, dtype=np.float64)[valid_mask]
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        if point_cloud.has_normals():
            normals = np.asarray(point_cloud.normals, dtype=np.float64)[valid_mask]
            point_cloud.normals = o3d.utility.Vector3dVector(normals)

    return summary


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from disk and validate that it contains points."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    if file_path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError(
            f"Unsupported file extension '{file_path.suffix}'. Expected .ply or .pcd."
        )

    point_cloud = o3d.io.read_point_cloud(str(file_path))
    if point_cloud.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {file_path}")

    summary = _sanitize_point_cloud(point_cloud)
    print(
        f"Point cloud validation: total={summary.total_points}, "
        f"removed_invalid={summary.removed_points} ({summary.removed_ratio * 100.0:.2f}%)"
    )
    if summary.removed_ratio > 0.05:
        raise ValueError(
            "More than 5% of the loaded points are invalid or extreme. "
            "Aborting to avoid contaminating the estimate."
        )
    if point_cloud.is_empty():
        raise ValueError(f"Loaded point cloud became empty after validation: {file_path}")

    return point_cloud
