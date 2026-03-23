"""Height-based filters for isolating stockpile points."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def height_filter(
    pcd: o3d.geometry.PointCloud,
    threshold: float = 0.1,
) -> tuple[o3d.geometry.PointCloud, float]:
    """Remove points that sit too close to the local ground reference."""
    if pcd.is_empty():
        raise ValueError("Cannot apply a height filter to an empty point cloud.")
    if threshold < 0:
        raise ValueError("height threshold must be non-negative.")

    points = np.asarray(pcd.points)
    z_min = float(points[:, 2].min())
    keep_indices = np.where(points[:, 2] > (z_min + threshold))[0]
    if keep_indices.size == 0:
        raise RuntimeError("Height filtering removed all points. Lower the height threshold.")

    return pcd.select_by_index(keep_indices.tolist()), z_min
