"""Entry point for robust sand volume estimation from a 3D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from src.clustering import cluster_objects, merge_clusters
from src.filters import build_ground_model, height_filter
from src.loader import load_point_cloud
from src.preprocess import estimate_mean_point_spacing, remove_noise, voxel_downsample
from src.roi import filter_by_polygon
from src.segmentation import remove_ground_plane
from src.visualization import pick_points_for_roi, show_clusters, show_pipeline_result
from src.volume import compute_bounding_box_volume, compute_validation_volumes, create_height_normalized_cloud


DEFAULT_INPUT = "WaterTankAndSand_20mm.ply"
DEFAULT_DOWNSAMPLE_VOXEL = 0.02
DEFAULT_VOLUME_VOXEL = 0.02
DEFAULT_DBSCAN_EPS = 0.12
DEFAULT_DBSCAN_MIN_POINTS = 100
DEFAULT_MIN_CLUSTER_SIZE = 200
DEFAULT_PLANE_THRESHOLD = 0.02
DEFAULT_HEIGHT_THRESHOLD = 0.0
DEFAULT_ROI_PADDING_XY = 0.5
DEFAULT_ROI_PADDING_Z = 0.5
ORIGINAL_VOLUME_REFERENCE_M3 = 1.8367


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate sand volume from a PLY or PCD point cloud using a guided ROI and robust filtering."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the input point cloud file (.ply or .pcd).")
    parser.add_argument("--downsample-voxel", type=float, default=DEFAULT_DOWNSAMPLE_VOXEL)
    parser.add_argument("--volume-voxel", type=float, default=DEFAULT_VOLUME_VOXEL)
    parser.add_argument("--dbscan-eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--dbscan-min-points", type=int, default=DEFAULT_DBSCAN_MIN_POINTS)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--plane-threshold", type=float, default=DEFAULT_PLANE_THRESHOLD)
    parser.add_argument("--height-threshold", type=float, default=DEFAULT_HEIGHT_THRESHOLD)
    parser.add_argument("--roi-padding-xy", type=float, default=DEFAULT_ROI_PADDING_XY)
    parser.add_argument("--roi-padding-z", type=float, default=DEFAULT_ROI_PADDING_Z)
    parser.add_argument("--cluster-index", type=int, default=None, help="Optional manual cluster override after clustering.")
    parser.add_argument("--no-vis", action="store_true", help="Disable Open3D result windows.")
    return parser.parse_args()


def _print_stage(name: str, pcd: o3d.geometry.PointCloud) -> None:
    print(f"{name}: {len(pcd.points)} points")


def _select_target_cloud(
    filtered_cloud: o3d.geometry.PointCloud,
    labels,
    clusters: list[o3d.geometry.PointCloud],
    requested_index: int | None,
) -> tuple[int, str, o3d.geometry.PointCloud, np.ndarray]:
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(clusters):
            raise IndexError(f"Requested cluster index {requested_index} is out of range for {len(clusters)} clusters.")
        selected_label = sorted([label for label in np.unique(labels) if label >= 0])[requested_index]
        selected_mask = labels == selected_label
        return requested_index, "manual_cluster", clusters[requested_index], selected_mask
    valid_mask = labels >= 0
    if np.any(valid_mask):
        return -1, "merged_roi_clusters", merge_clusters(clusters), valid_mask
    return -1, "full_filtered_roi", filtered_cloud, np.ones(len(filtered_cloud.points), dtype=bool)


def _score_confidence(
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    print(f"Loading point cloud from: {input_path}")
    point_cloud = load_point_cloud(input_path)
    _print_stage("Loaded cloud", point_cloud)

    picked_points = pick_points_for_roi(point_cloud)
    roi_cloud, polygon = filter_by_polygon(
        point_cloud,
        picked_points,
        padding_xy=args.roi_padding_xy,
        padding_z=args.roi_padding_z,
    )
    _print_stage("ROI-filtered cloud", roi_cloud)
    print(f"ROI polygon vertices: {len(polygon)}")

    working_voxel = max(args.downsample_voxel, estimate_mean_point_spacing(roi_cloud) * 1.5)
    downsampled_cloud = voxel_downsample(roi_cloud, voxel_size=working_voxel)
    _print_stage("Downsampled cloud", downsampled_cloud)

    denoised_cloud = remove_noise(downsampled_cloud)
    _print_stage("Denoised cloud", denoised_cloud)

    ground_cloud, non_ground_cloud, plane_model, _ = remove_ground_plane(
        denoised_cloud,
        distance_threshold=args.plane_threshold,
    )
    _print_stage("Ground cloud", ground_cloud)
    _print_stage("Non-ground cloud", non_ground_cloud)
    print(
        "Ground plane model: "
        f"{plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0"
    )

    ground_model = build_ground_model(ground_cloud)
    filtered_object_cloud, filtered_heights, ground_rmse = height_filter(
        non_ground_cloud,
        ground_model=ground_model,
        threshold=args.height_threshold,
    )
    _print_stage("Height-filtered object cloud", filtered_object_cloud)
    print(f"Ground fitting RMSE: {ground_rmse:.4f} m")

    labels, clusters, summaries, clustering_params = cluster_objects(
        filtered_object_cloud,
        eps=args.dbscan_eps,
        min_points=args.dbscan_min_points,
        min_cluster_size=args.min_cluster_size,
    )
    print(f"Number of clusters: {len(clusters)}")
    print("Cluster sizes:", [summary.size for summary in summaries])
    print(f"Adaptive DBSCAN eps: {clustering_params.eps:.4f}")
    print(f"Adaptive DBSCAN min_points: {clustering_params.min_points}")

    selected_index, selected_strategy, selected_cloud, selected_mask = _select_target_cloud(
        filtered_object_cloud,
        labels,
        clusters,
        args.cluster_index,
    )
    print(f"Selection strategy: {selected_strategy}")
    print(f"Selected cluster index: {selected_index}")
    print(f"Selected cloud size: {len(selected_cloud.points)}")
    print(f"Points removed by clustering: {max(len(filtered_object_cloud.points) - sum(summary.size for summary in summaries), 0)}")

    selected_heights = filtered_heights[selected_mask]
    normalized_selected_cloud = create_height_normalized_cloud(selected_cloud, selected_heights)
    bbox_volume, bbox = compute_bounding_box_volume(normalized_selected_cloud)
    validation = compute_validation_volumes(normalized_selected_cloud, fallback_voxel_size=args.volume_voxel)

    roi_completeness = float(len(selected_cloud.points) / max(len(filtered_object_cloud.points), 1))
    selected_spacing = estimate_mean_point_spacing(selected_cloud)
    warnings = list(validation.warnings)
    if ground_rmse > 0.05:
        warnings.append("Ground fit RMSE is above 0.05 m; ground estimate is unreliable.")
    confidence, error_estimate_percent = _score_confidence(
        ground_rmse=ground_rmse,
        empty_voxel_percent=validation.voxel_metrics.empty_ratio * 100.0,
        roi_completeness=roi_completeness,
        spacing=selected_spacing,
        warnings=warnings,
    )

    print("\nDebug Metrics")
    print("-------------")
    print(f"Mean point spacing: {selected_spacing:.4f} m")
    print(f"Adaptive voxel size: {validation.adaptive_voxel_size:.4f} m")
    print(f"ROI completeness: {roi_completeness:.3f}")
    print(f"% empty voxels: {validation.voxel_metrics.empty_ratio * 100.0:.2f}")
    print(f"Ground plane error: {ground_rmse:.4f} m")
    print(f"ROI size: {len(roi_cloud.points)} points")

    print("\nVolume Validation")
    print("-----------------")
    print(f"Bounding Box Volume (comparison): {bbox_volume:.4f} m³")
    print(f"Binary Voxel Volume:             {validation.voxel_volume_m3:.4f} m³")
    print(f"Weighted Voxel Volume:           {validation.weighted_voxel_volume_m3:.4f} m³")
    print(f"Height Map Volume:               {validation.height_map_volume_m3:.4f} m³")
    print(f"Mesh Volume:                     {validation.mesh_volume_m3:.4f} m³")
    print(f"Final Estimated Sand Volume:     {validation.final_volume_m3:.4f} m³")
    print(f"Method Used:                     {validation.method_used}")
    print(f"Difference from 1.8367 m³:       {validation.final_volume_m3 - ORIGINAL_VOLUME_REFERENCE_M3:+.4f} m³")
    print(f"Confidence:                      {confidence}")
    print(f"Estimated error:                 {error_estimate_percent:.2f}%")
    print(f"Warnings:                        {warnings if warnings else 'none'}")

    if not args.no_vis:
        show_clusters(filtered_object_cloud, labels)
        show_pipeline_result(ground_cloud, filtered_object_cloud, selected_cloud, bbox)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
