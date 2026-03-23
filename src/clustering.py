"""Point cloud clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from src.preprocess import estimate_mean_point_spacing


@dataclass(slots=True)
class ClusterSummary:
    label: int
    size: int


@dataclass(slots=True)
class ClusteringParameters:
    eps: float
    min_points: int
    spacing: float


def resolve_clustering_parameters(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.12,
    min_points: int = 100,
) -> ClusteringParameters:
    """Adapt DBSCAN parameters to local point spacing to reduce data loss."""
    spacing = estimate_mean_point_spacing(pcd)
    adaptive_eps = max(eps, spacing * 3.5)
    adaptive_min_points = max(20, min(min_points, int(round(2.0 / max(spacing, 1e-3)))))
    return ClusteringParameters(
        eps=float(adaptive_eps),
        min_points=int(adaptive_min_points),
        spacing=float(spacing),
    )


def cluster_objects(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.12,
    min_points: int = 100,
    min_cluster_size: int = 200,
) -> tuple[np.ndarray, list[o3d.geometry.PointCloud], list[ClusterSummary], ClusteringParameters]:
    """Cluster points with spacing-adaptive DBSCAN, ignore noise, and remove tiny clusters."""
    if pcd.is_empty():
        raise ValueError("Cannot cluster an empty point cloud.")

    params = resolve_clustering_parameters(pcd, eps=eps, min_points=min_points)
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=params.eps,
            min_points=params.min_points,
            print_progress=False,
        )
    )

    valid_labels = [label for label in np.unique(labels) if label >= 0]
    clusters: list[o3d.geometry.PointCloud] = []
    summaries: list[ClusterSummary] = []

    for label in valid_labels:
        indices = np.where(labels == label)[0]
        size = int(indices.size)
        if size < min_cluster_size:
            labels[indices] = -1
            continue
        clusters.append(pcd.select_by_index(indices.tolist()))
        summaries.append(ClusterSummary(label=int(label), size=size))

    return labels, clusters, summaries, params


def merge_clusters(clusters: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge multiple valid clusters into one cloud for ROI-driven selection."""
    if not clusters:
        raise ValueError("No clusters were provided for merging.")

    merged_points = []
    merged_colors = []
    for cluster in clusters:
        if cluster.is_empty():
            continue
        merged_points.append(np.asarray(cluster.points))
        if cluster.has_colors():
            merged_colors.append(np.asarray(cluster.colors))

    if not merged_points:
        raise ValueError("All provided clusters were empty.")

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(merged_points))
    if merged_colors and len(merged_colors) == len(merged_points):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
    return merged
