"""Backend service helpers for the browser-based volume estimation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from src.clustering import cluster_objects, merge_clusters
from src.filters import build_ground_model, height_filter
from src.loader import load_point_cloud
from src.preprocess import estimate_mean_point_spacing, remove_noise, voxel_downsample
from src.roi import compute_cuboid_stats, filter_by_bounds, filter_by_cuboid, filter_by_polygon
from src.segmentation import remove_ground_plane
from src.volume import (
    compute_bounding_box_volume,
    compute_validation_volumes,
    create_height_normalized_cloud,
    voxel_grid_to_point_cloud,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = {".ply", ".pcd"}
MAX_PREVIEW_POINTS = 40000
MAX_RESULT_POINTS = 30000
EMPTY_CLOUD_PAYLOAD = {
    "positions": [],
    "colors": [],
    "point_count": 0,
    "bbox": {"min": [0, 0, 0], "max": [0, 0, 0]},
}


@dataclass(slots=True)
class AnalysisSummary:
    selected_cluster_index: int
    selected_strategy: str
    cluster_count: int
    cluster_sizes: list[int]
    total_points_loaded: int
    roi_points: int
    downsampled_points: int
    denoised_points: int
    ground_points: int
    object_points: int
    selected_points: int
    points_removed_by_clustering: int
    mean_point_spacing: float
    ground_rmse: float
    roi_completeness: float
    voxel_count: int
    method_used: str
    confidence: str
    error_estimate_percent: float
    warnings: list[str]
    plane_model: list[float]
    bbox_volume_m3: float
    final_volume_m3: float
    binary_voxel_volume_m3: float
    weighted_voxel_volume_m3: float
    height_map_volume_m3: float
    mesh_volume_m3: float
    adaptive_voxel_size: float
    empty_voxel_percent: float
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
        return dict(EMPTY_CLOUD_PAYLOAD)

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


def _score_confidence(
    *,
    ground_rmse: float,
    empty_voxel_percent: float,
    roi_completeness: float,
    spacing: float,
    warnings: list[str],
) -> tuple[str, float]:
    estimated_error = (
        min(ground_rmse * 120.0, 12.0)
        + min(empty_voxel_percent * 0.15, 12.0)
        + max(0.0, (0.9 - roi_completeness) * 40.0)
        + min(spacing * 150.0, 8.0)
        + min(len(warnings) * 2.0, 8.0)
    )
    if estimated_error <= 3.0:
        return "high", float(estimated_error)
    if estimated_error <= 7.0:
        return "medium", float(estimated_error)
    return "low", float(estimated_error)


def list_point_cloud_files() -> list[str]:
    files = []
    for path in WORKSPACE_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(str(path.relative_to(WORKSPACE_ROOT)))
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


def get_cuboid_stats(*, input_path: str, selection_cuboid: dict[str, object]) -> dict[str, object]:
    source_path = _resolve_workspace_file(input_path)
    point_cloud = load_point_cloud(source_path)
    stats = compute_cuboid_stats(point_cloud, selection_cuboid)
    return {
        **stats,
        "selected_points": int(stats["selected_points"]),
        "excluded_points": int(stats["excluded_points"]),
    }


def _select_target_cloud(
    filtered_cloud: o3d.geometry.PointCloud,
    labels: np.ndarray,
    clusters: list[o3d.geometry.PointCloud],
    cluster_index: int | None,
    selection_mode: str,
) -> tuple[int, str, o3d.geometry.PointCloud, np.ndarray]:
    if cluster_index is not None:
        if cluster_index < 0 or cluster_index >= len(clusters):
            raise IndexError(f"Requested cluster index {cluster_index} is out of range for {len(clusters)} clusters.")
        selected_label = sorted([label for label in np.unique(labels) if label >= 0])[cluster_index]
        selected_mask = labels == selected_label
        return cluster_index, "manual_cluster", clusters[cluster_index], selected_mask

    if selection_mode in {"box", "cuboid"}:
        return -1, f"{selection_mode}_full_roi", filtered_cloud, np.ones(len(filtered_cloud.points), dtype=bool)

    valid_mask = labels >= 0
    if np.any(valid_mask):
        return -1, "merged_roi_clusters", merge_clusters(clusters), valid_mask
    return -1, "full_filtered_roi", filtered_cloud, np.ones(len(filtered_cloud.points), dtype=bool)


def analyze_selected_region(
    *,
    input_path: str,
    picked_points: list[list[float]],
    selection_mode: str = "polygon",
    selection_bounds: dict[str, list[float]] | None = None,
    selection_cuboid: dict[str, object] | None = None,
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
    if selection_mode != "cuboid" and len(picked_points) < 3:
        raise ValueError("Pick at least 3 points on the target object region before analysis.")

    source_path = _resolve_workspace_file(input_path)
    point_cloud = load_point_cloud(source_path)
    seed_points = np.asarray(picked_points, dtype=float) if picked_points else np.empty((0, 3), dtype=float)
    cuboid_stats: dict[str, object] | None = None

    if selection_mode == "cuboid":
        if selection_cuboid is None:
            raise ValueError("Cuboid selection requires cuboid parameters.")
        roi_cloud, cuboid_stats = filter_by_cuboid(point_cloud, selection_cuboid)
    elif selection_mode == "box":
        if not selection_bounds or "min" not in selection_bounds or "max" not in selection_bounds:
            raise ValueError("Box selection requires selection bounds.")
        roi_cloud = filter_by_bounds(
            point_cloud,
            np.asarray(selection_bounds["min"], dtype=float),
            np.asarray(selection_bounds["max"], dtype=float),
            padding_xy=roi_padding_xy,
            padding_z=roi_padding_z,
            full_height=True,
        )
    else:
        roi_cloud, _ = filter_by_polygon(
            point_cloud,
            seed_points,
            padding_xy=roi_padding_xy,
            padding_z=roi_padding_z,
        )

    working_voxel = max(downsample_voxel, estimate_mean_point_spacing(roi_cloud) * 1.5)
    downsampled_cloud = voxel_downsample(roi_cloud, voxel_size=working_voxel)
    denoised_cloud = remove_noise(downsampled_cloud)

    direct_object_mode = selection_mode == "cuboid"
    pre_selection_warnings: list[str] = []
    if cuboid_stats is not None and selection_cuboid and selection_cuboid.get("snap_to_ground"):
        pre_selection_warnings.append("Cuboid bottom snapped relative to the detected local ground.")

    if direct_object_mode:
        ground_cloud = o3d.geometry.PointCloud()
        filtered_cloud = denoised_cloud
        filtered_heights = np.asarray(filtered_cloud.points)[:, 2] - float(np.asarray(filtered_cloud.points)[:, 2].min())
        ground_rmse = 0.0
        plane_model = [0.0, 0.0, 1.0, -float(np.asarray(filtered_cloud.points)[:, 2].min())]
        labels = np.zeros(len(filtered_cloud.points), dtype=int)
        clusters = [filtered_cloud]
        summaries = []
    else:
        try:
            ground_cloud, non_ground_cloud, plane_model, _ = remove_ground_plane(
                denoised_cloud,
                distance_threshold=plane_threshold,
            )
            ground_model = build_ground_model(ground_cloud)
            filtered_cloud, filtered_heights, ground_rmse = height_filter(
                non_ground_cloud,
                ground_model=ground_model,
                threshold=height_threshold,
            )
        except RuntimeError as exc:
            if "sufficiently horizontal" not in str(exc):
                raise
            pre_selection_warnings.append(
                "No horizontal ground plane detected; analyzing the selected object geometry directly."
            )
            direct_object_mode = True
            ground_cloud = o3d.geometry.PointCloud()
            filtered_cloud = denoised_cloud
            filtered_heights = np.asarray(filtered_cloud.points)[:, 2] - float(np.asarray(filtered_cloud.points)[:, 2].min())
            ground_rmse = 0.0
            plane_model = [0.0, 0.0, 1.0, -float(np.asarray(filtered_cloud.points)[:, 2].min())]

        if direct_object_mode:
            labels = np.zeros(len(filtered_cloud.points), dtype=int)
            clusters = [filtered_cloud]
            summaries = []
        else:
            labels, clusters, summaries, _ = cluster_objects(
                filtered_cloud,
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                min_cluster_size=min_cluster_size,
            )

    selected_index, selected_strategy, selected_cloud, selected_mask = _select_target_cloud(
        filtered_cloud,
        labels,
        clusters,
        cluster_index,
        selection_mode,
    )

    selected_heights = filtered_heights[selected_mask]
    analysis_cloud = selected_cloud if direct_object_mode else create_height_normalized_cloud(selected_cloud, selected_heights)
    validation = compute_validation_volumes(analysis_cloud, fallback_voxel_size=volume_voxel)
    bbox_volume, bbox = compute_bounding_box_volume(analysis_cloud)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(analysis_cloud, validation.adaptive_voxel_size)
    voxel_cloud = voxel_grid_to_point_cloud(voxel_grid)

    points_removed_by_clustering = max(len(filtered_cloud.points) - sum(summary.size for summary in summaries), 0)
    roi_completeness = float(len(selected_cloud.points) / max(len(filtered_cloud.points), 1))
    selected_spacing = estimate_mean_point_spacing(selected_cloud)
    warnings = list(pre_selection_warnings) + list(validation.warnings)
    if (not direct_object_mode) and ground_rmse > 0.05:
        warnings.append("Ground fit RMSE is above 0.05 m; ground estimate is unreliable.")
    confidence, error_estimate_percent = _score_confidence(
        ground_rmse=ground_rmse,
        empty_voxel_percent=validation.voxel_metrics.empty_ratio * 100.0,
        roi_completeness=roi_completeness,
        spacing=selected_spacing,
        warnings=warnings,
    )

    return AnalysisSummary(
        selected_cluster_index=selected_index,
        selected_strategy=selected_strategy,
        cluster_count=len(clusters),
        cluster_sizes=[summary.size for summary in summaries],
        total_points_loaded=len(point_cloud.points),
        roi_points=len(roi_cloud.points),
        downsampled_points=len(downsampled_cloud.points),
        denoised_points=len(denoised_cloud.points),
        ground_points=len(ground_cloud.points),
        object_points=len(filtered_cloud.points),
        selected_points=len(selected_cloud.points),
        points_removed_by_clustering=points_removed_by_clustering,
        mean_point_spacing=selected_spacing,
        ground_rmse=ground_rmse,
        roi_completeness=roi_completeness,
        voxel_count=validation.voxel_metrics.occupied_voxels,
        method_used=validation.method_used,
        confidence=confidence,
        error_estimate_percent=error_estimate_percent,
        warnings=warnings,
        plane_model=plane_model,
        bbox_volume_m3=float(bbox_volume),
        final_volume_m3=validation.final_volume_m3,
        binary_voxel_volume_m3=validation.voxel_volume_m3,
        weighted_voxel_volume_m3=validation.weighted_voxel_volume_m3,
        height_map_volume_m3=validation.height_map_volume_m3,
        mesh_volume_m3=validation.mesh_volume_m3,
        adaptive_voxel_size=validation.adaptive_voxel_size,
        empty_voxel_percent=validation.voxel_metrics.empty_ratio * 100.0,
        bbox_min=bbox.get_min_bound().tolist(),
        bbox_max=bbox.get_max_bound().tolist(),
        ground_cloud_payload=_serialize_point_cloud(ground_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.1, 0.75, 0.2)),
        filtered_cloud_payload=_serialize_point_cloud(filtered_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.9, 0.2, 0.2)),
        selected_cloud_payload=_serialize_point_cloud(selected_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.15, 0.4, 0.95)),
        voxel_cloud_payload=_serialize_point_cloud(voxel_cloud, max_points=MAX_RESULT_POINTS, default_color=(0.12, 0.86, 1.0)),
    )
