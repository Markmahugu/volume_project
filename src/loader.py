"""Utilities for loading point cloud files."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d


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

    return point_cloud
