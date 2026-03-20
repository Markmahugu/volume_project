"""Preprocessing utilities for point cloud filtering."""

from __future__ import annotations

import open3d as o3d


def voxel_downsample(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    """Reduce point density while preserving the overall object geometry."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")

    if pcd.is_empty():
        raise ValueError("Cannot downsample an empty point cloud.")

    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_noise(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Remove sparse outliers using statistical filtering."""
    if pcd.is_empty():
        raise ValueError("Cannot remove noise from an empty point cloud.")

    filtered_cloud, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered_cloud
