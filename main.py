"""Entry point for robust sand volume estimation from a 3D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from src.clustering import cluster_objects, select_largest_cluster
from src.filters import height_filter
from src.loader import load_point_cloud
from src.preprocess import remove_noise, voxel_downsample
from src.roi import filter_by_polygon
from src.segmentation import remove_ground_plane
from src.visualization import pick_points_for_roi, show_clusters, show_pipeline_result
from src.volume import compute_bounding_box_volume, compute_voxel_volume


DEFAULT_INPUT = "WaterTankAndSand_20mm.ply"
DEFAULT_DOWNSAMPLE_VOXEL = 0.02
DEFAULT_VOLUME_VOXEL = 0.02
DEFAULT_DBSCAN_EPS = 0.12
DEFAULT_DBSCAN_MIN_POINTS = 100
DEFAULT_MIN_CLUSTER_SIZE = 200
DEFAULT_PLANE_THRESHOLD = 0.02
DEFAULT_HEIGHT_THRESHOLD = 0.10
DEFAULT_ROI_PADDING_XY = 0.5
DEFAULT_ROI_PADDING_Z = 0.5


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


def _select_cluster(
    clusters: list[o3d.geometry.PointCloud],
    requested_index: int | None,
) -> tuple[int, o3d.geometry.PointCloud]:
    if not clusters:
        raise RuntimeError("No valid clusters remain after filtering.")
    if requested_index is None:
        return select_largest_cluster(clusters)
    if requested_index < 0 or requested_index >= len(clusters):
        raise IndexError(f"Requested cluster index {requested_index} is out of range for {len(clusters)} clusters.")
    return requested_index, clusters[requested_index]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    print(f"Loading point cloud from: {input_path}")
    point_cloud = load_point_cloud(input_path)
    _print_stage("Loaded cloud", point_cloud)

    # ROI first: user picks define the XY footprint so downstream filters only see the target pile neighborhood.
    picked_points = pick_points_for_roi(point_cloud)
    roi_cloud, polygon = filter_by_polygon(
        point_cloud,
        picked_points,
        padding_xy=args.roi_padding_xy,
        padding_z=args.roi_padding_z,
    )
    _print_stage("ROI-filtered cloud", roi_cloud)
    print(f"ROI polygon vertices: {len(polygon)}")

    # Downsampling stabilizes RANSAC and DBSCAN while keeping the stockpile geometry.
    downsampled_cloud = voxel_downsample(roi_cloud, voxel_size=args.downsample_voxel)
    _print_stage("Downsampled cloud", downsampled_cloud)

    # Statistical denoising removes isolated scan artifacts that would otherwise form false clusters.
    denoised_cloud = remove_noise(downsampled_cloud)
    _print_stage("Denoised cloud", denoised_cloud)

    # Stronger plane removal isolates the stockpile from the local ground surface.
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

    # Height filtering removes low residual ground fragments that survive plane segmentation.
    filtered_object_cloud, z_min = height_filter(non_ground_cloud, threshold=args.height_threshold)
    _print_stage("Height-filtered object cloud", filtered_object_cloud)
    print(f"Height filter reference z_min: {z_min:.4f} m")
    print(f"Height threshold: {args.height_threshold:.4f} m")

    # DBSCAN groups dense pile points and suppresses sparse clutter/noise.
    labels, clusters, summaries = cluster_objects(
        filtered_object_cloud,
        eps=args.dbscan_eps,
        min_points=args.dbscan_min_points,
        min_cluster_size=args.min_cluster_size,
    )
    print(f"Number of clusters: {len(clusters)}")
    print("Cluster sizes:", [summary.size for summary in summaries])

    selected_index, selected_cluster = _select_cluster(clusters, args.cluster_index)
    print(f"Selected cluster index: {selected_index}")
    print(f"Selected cluster size: {len(selected_cluster.points)}")
    if args.cluster_index is not None:
        print("Manual cluster selection override applied.")
    else:
        print("Largest cluster selected automatically.")

    bbox_volume, bbox = compute_bounding_box_volume(selected_cluster)
    voxel_volume, _ = compute_voxel_volume(selected_cluster, voxel_size=args.volume_voxel)

    print("\nVolume Estimation Results")
    print("-------------------------")
    print(f"Bounding Box Volume (comparison): {bbox_volume:.2f} m³")
    print(f"Voxel Volume (FINAL estimate):    {voxel_volume:.2f} m³")
    print(f"Final Estimated Sand Volume: {voxel_volume:.2f} m³")

    if not args.no_vis:
        show_clusters(filtered_object_cloud, labels)
        show_pipeline_result(ground_cloud, filtered_object_cloud, selected_cluster, bbox)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
