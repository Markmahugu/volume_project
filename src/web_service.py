"""Backend service helpers for the browser-based volume estimation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from src.clustering import cluster_objects, select_largest_cluster
from src.filters import height_filter
from src.loader import load_point_cloud
from src.preprocess import remove_noise, voxel_downsample
from src.roi import filter_by_polygon
from src.segmentation import remove_ground_plane
from src.volume import compute_bounding_box_volume, compute_voxel_volume, voxel_grid_to_point_cloud

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = {".ply", ".pcd"}
MAX_PREVIEW_POINTS = 40000
MAX_RESULT_POINTS = 30000


@dataclass(slots=True)
class AnalysisSummary:
    selected_cluster_index: int
    cluster_count: int
    cluster_sizes: list[int]
    total_points_loaded: int
    roi_points: int
    denoised_points: int
    ground_points: int
    object_points: int
    selected_points: int
    voxel_count: int
    z_min: float
    plane_model: list[float]
    bbox_volume_m3: float
    voxel_volume_m3: float
    bbox_min: list[float]
    bbox_max: list[float]
    ground_cloud_payload: dict[str, object]
    filtered_cloud_payload: dict[str, object]
    selected_cloud_payload: dict[str, object]
    voxel_cloud_payload: dict[str, object]


def _resolve_workspace_file(input_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / input_path).resolve()
    if not str(candidate).startswith(str(WORKSPACE_ROOT.resolve())):
        raise ValueError("Input path must stay within the workspace.")
    if not candidate.exists():
        raise FileNotFoundError(f"Point cloud file not found: {candidate}")
    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .ply and .pcd point cloud files are supported.")
    return candidate


def _sample_point_cloud(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    if len(pcd.points) <= max_points:
        return o3d.geometry.PointCloud(pcd)
    return pcd.random_down_sample(max_points / len(pcd.points))


def _serialize_point_cloud(
    pcd: o3d.geometry.PointCloud,
    *,
    max_points: int,
    default_color: tuple[float, float, float],
) -> dict[str, object]:
    sampled = _sample_point_cloud(pcd, max_points=max_points)
    points = np.asarray(sampled.points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError("Cannot serialize an empty point cloud.")

    if sampled.has_colors():
        colors = np.asarray(sampled.colors, dtype=np.float32)
    else:
        colors = np.tile(np.asarray(default_color, dtype=np.float32), (points.shape[0], 1))

    bbox = sampled.get_axis_aligned_bounding_box()
    return {
        "positions": points.reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
        "point_count": int(points.shape[0]),
        "bbox": {
            "min": bbox.get_min_bound().tolist(),
            "max": bbox.get_max_bound().tolist(),
        },
    }


def list_point_cloud_files() -> list[str]:
    files = []
    for path in WORKSPACE_ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path.name)
    return sorted(files)


def get_preview_payload(input_path: str, voxel_size: float = 0.05) -> dict[str, object]:
    source_path = _resolve_workspace_file(input_path)
    point_cloud = load_point_cloud(source_path)
    preview_cloud = voxel_downsample(point_cloud, voxel_size=voxel_size)
    if preview_cloud.is_empty():
        raise RuntimeError("Preview generation produced an empty point cloud.")

    payload = _serialize_point_cloud(preview_cloud, max_points=MAX_PREVIEW_POINTS, default_color=(0.85, 0.85, 0.85))
    payload["input_path"] = input_path
    payload["source_points"] = len(point_cloud.points)
    return payload


def _select_cluster(
    clusters: list[o3d.geometry.PointCloud],
    cluster_index: int | None,
) -> tuple[int, o3d.geometry.PointCloud]:
    if not clusters:
        raise RuntimeError("No valid clusters remain after filtering.")
    if cluster_index is None:
        return select_largest_cluster(clusters)
    if cluster_index < 0 or cluster_index >= len(clusters):
        raise IndexError(f"Requested cluster index {cluster_index} is out of range for {len(clusters)} clusters.")
    return cluster_index, clusters[cluster_index]


def analyze_selected_region(
    *,
    input_path: str,
    picked_points: list[list[float]],
    downsample_voxel: float,
    volume_voxel: float,
    dbscan_eps: float,
    dbscan_min_points: int,
    min_cluster_size: int,
    plane_threshold: float,
    height_threshold: float,
    roi_padding_xy: float,
    roi_padding_z: float,
    cluster_index: int | None = None,
) -> AnalysisSummary:
    if len(picked_points) < 3:
        raise ValueError("Pick at least 3 points on the sand region before analysis.")

    source_path = _resolve_workspace_file(input_path)
    seed_points = np.asarray(picked_points, dtype=float)

    point_cloud = load_point_cloud(source_path)
    roi_cloud, _ = filter_by_polygon(
        point_cloud,
        seed_points,
        padding_xy=roi_padding_xy,
        padding_z=roi_padding_z,
    )
    downsampled_cloud = voxel_downsample(roi_cloud, voxel_size=downsample_voxel)
    denoised_cloud = remove_noise(downsampled_cloud)
    ground_cloud, non_ground_cloud, plane_model, _ = remove_ground_plane(
        denoised_cloud,
        distance_threshold=plane_threshold,
    )
    filtered_cloud, z_min = height_filter(non_ground_cloud, threshold=height_threshold)
    _, clusters, summaries = cluster_objects(
        filtered_cloud,
        eps=dbscan_eps,
        min_points=dbscan_min_points,
        min_cluster_size=min_cluster_size,
    )
    selected_index, selected_cluster = _select_cluster(clusters, cluster_index)

    bbox_volume, bbox = compute_bounding_box_volume(selected_cluster)
    voxel_volume, voxel_grid = compute_voxel_volume(selected_cluster, voxel_size=volume_voxel)
    voxel_cloud = voxel_grid_to_point_cloud(voxel_grid)

    return AnalysisSummary(
        selected_cluster_index=selected_index,
        cluster_count=len(clusters),
        cluster_sizes=[summary.size for summary in summaries],
        total_points_loaded=len(point_cloud.points),
        roi_points=len(roi_cloud.points),
        denoised_points=len(denoised_cloud.points),
        ground_points=len(ground_cloud.points),
        object_points=len(filtered_cloud.points),
        selected_points=len(selected_cluster.points),
        voxel_count=len(voxel_grid.get_voxels()),
        z_min=float(z_min),
        plane_model=plane_model,
        bbox_volume_m3=float(bbox_volume),
        voxel_volume_m3=float(voxel_volume),
        bbox_min=bbox.get_min_bound().tolist(),
        bbox_max=bbox.get_max_bound().tolist(),
        ground_cloud_payload=_serialize_point_cloud(ground_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.1, 0.75, 0.2)),
        filtered_cloud_payload=_serialize_point_cloud(filtered_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.9, 0.2, 0.2)),
        selected_cloud_payload=_serialize_point_cloud(selected_cluster, max_points=MAX_RESULT_POINTS, default_color=(0.15, 0.4, 0.95)),
        voxel_cloud_payload=_serialize_point_cloud(voxel_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.12, 0.86, 1.0)),
    )
