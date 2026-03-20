"""Entry point for volume estimation from a 3D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering import cluster_objects
from src.loader import load_point_cloud
from src.preprocess import remove_noise, voxel_downsample
from src.segmentation import remove_ground_plane
from src.visualization import show_clusters, show_with_bbox
from src.volume import compute_bounding_box_volume, compute_voxel_volume


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the volume estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="Estimate object volume from a PLY or PCD point cloud using Open3D."
    )
    parser.add_argument(
        "--input",
        default="WaterTankAndSand_20mm.ply",
        help="Path to the input point cloud file (.ply or .pcd).",
    )
    parser.add_argument(
        "--downsample-voxel",
        type=float,
        default=0.02,
        help="Voxel size used for downsampling during preprocessing.",
    )
    parser.add_argument(
        "--volume-voxel",
        type=float,
        default=0.02,
        help="Voxel size used for the voxel-based volume estimate.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.08,
        help="Neighborhood radius for DBSCAN clustering.",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=40,
        help="Minimum number of points required to form a DBSCAN cluster.",
    )
    parser.add_argument(
        "--plane-threshold",
        type=float,
        default=0.03,
        help="Distance threshold used by RANSAC plane segmentation.",
    )
    parser.add_argument(
        "--cluster-index",
        type=int,
        default=None,
        help="Optional cluster index to select. Defaults to the largest cluster.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable Open3D visualization windows.",
    )
    return parser.parse_args()


def select_cluster(
    clusters: list,
    requested_index: int | None,
):
    """Select a cluster by explicit index or choose the largest cluster."""
    if not clusters:
        raise RuntimeError("No object clusters were detected after ground removal.")

    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(clusters):
            raise IndexError(
                f"Requested cluster index {requested_index} is out of range for {len(clusters)} clusters."
            )
        return requested_index, clusters[requested_index]

    cluster_sizes = [len(cluster.points) for cluster in clusters]
    largest_index = max(range(len(clusters)), key=lambda idx: cluster_sizes[idx])
    return largest_index, clusters[largest_index]


def main() -> None:
    """Execute the full point-cloud volume estimation pipeline."""
    args = parse_args()
    input_path = Path(args.input)

    print(f"Loading point cloud from: {input_path}")
    point_cloud = load_point_cloud(input_path)
    print(f"Loaded {len(point_cloud.points)} points.")

    # Step 1: Reduce density to speed up downstream processing.
    downsampled_cloud = voxel_downsample(point_cloud, voxel_size=args.downsample_voxel)
    print(f"Downsampled point cloud to {len(downsampled_cloud.points)} points.")

    # Step 2: Remove statistical outliers from the scan.
    filtered_cloud = remove_noise(downsampled_cloud)
    print(f"After noise removal: {len(filtered_cloud.points)} points.")

    # Step 3: Remove the dominant ground plane with RANSAC.
    object_cloud, plane_model, _ = remove_ground_plane(
        filtered_cloud,
        distance_threshold=args.plane_threshold,
    )
    print(f"Removed ground plane. Remaining object points: {len(object_cloud.points)}")
    print(
        "Ground plane model: "
        f"{plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0"
    )

    # Step 4: Cluster the non-ground points to isolate individual objects.
    labels, clusters = cluster_objects(
        object_cloud,
        eps=args.dbscan_eps,
        min_points=args.dbscan_min_points,
    )
    print(f"Detected {len(clusters)} cluster(s).")

    selected_index, selected_cluster = select_cluster(clusters, args.cluster_index)
    print(f"Selected cluster {selected_index} with {len(selected_cluster.points)} points.")

    # Step 5: Estimate object volume with two different approximations.
    bbox_volume, bbox = compute_bounding_box_volume(selected_cluster)
    voxel_volume, _ = compute_voxel_volume(
        selected_cluster,
        voxel_size=args.volume_voxel,
    )

    print("\nVolume Estimation Results")
    print("-------------------------")
    print(f"Axis-aligned bounding box volume: {bbox_volume:.4f} m^3")
    print(f"Voxel-based occupied volume:      {voxel_volume:.4f} m^3")
    print(f"Estimated Volume: {voxel_volume:.4f} m^3")

    if not args.no_vis:
        show_clusters(object_cloud, labels)
        show_with_bbox(selected_cluster, bbox)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
