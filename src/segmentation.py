"""Ground segmentation helpers."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 2500,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, list[float], list[int]]:
    """Segment the dominant ground plane and return ground plus non-ground clouds."""
    if pcd.is_empty():
        raise ValueError("Cannot segment an empty point cloud.")

    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    ground_cloud = pcd.select_by_index(inlier_indices)
    non_ground_cloud = pcd.select_by_index(inlier_indices, invert=True)
    if non_ground_cloud.is_empty():
        raise RuntimeError("Ground removal removed all points from the point cloud.")

    normal = np.asarray(plane_model[:3], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    if abs(normal[2]) < 0.7:
        raise RuntimeError("Dominant plane is not sufficiently horizontal to be treated as ground.")

    return ground_cloud, non_ground_cloud, [float(value) for value in plane_model], inlier_indices
