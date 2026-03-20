"""Ground segmentation helpers."""

from __future__ import annotations

import open3d as o3d


def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[o3d.geometry.PointCloud, list[float], list[int]]:
    """Segment the dominant plane and return the non-plane points."""
    if pcd.is_empty():
        raise ValueError("Cannot segment an empty point cloud.")

    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    object_cloud = pcd.select_by_index(inlier_indices, invert=True)
    if object_cloud.is_empty():
        raise RuntimeError("Ground removal removed all points from the point cloud.")

    return object_cloud, plane_model, inlier_indices
