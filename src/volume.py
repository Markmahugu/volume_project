"""Volume estimation methods."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def compute_bounding_box_volume(
    pcd: o3d.geometry.PointCloud,
) -> tuple[float, o3d.geometry.AxisAlignedBoundingBox]:
    """Estimate volume using an axis-aligned bounding box."""
    if pcd.is_empty():
        raise ValueError("Cannot compute volume for an empty point cloud.")

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=float)
    volume = float(np.prod(extent))
    return volume, bbox


def compute_voxel_volume(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> tuple[float, o3d.geometry.VoxelGrid]:
    """Estimate volume from occupied voxels in a voxel grid."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")

    if pcd.is_empty():
        raise ValueError("Cannot compute volume for an empty point cloud.")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size,
    )
    occupied_voxels = len(voxel_grid.get_voxels())
    volume = occupied_voxels * (voxel_size ** 3)
    return float(volume), voxel_grid
