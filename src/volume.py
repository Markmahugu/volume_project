"""Volume estimation methods."""

from __future__ import annotations

import numpy as np
import open3d as o3d


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


def compute_voxel_volume(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[float, o3d.geometry.VoxelGrid]:
    """Estimate volume from occupied voxels. This is the primary estimate."""
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
