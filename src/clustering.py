"""Point cloud clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(slots=True)
class ClusterSummary:
    label: int
    size: int


def cluster_objects(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.12,
    min_points: int = 100,
    min_cluster_size: int = 200,
) -> tuple[np.ndarray, list[o3d.geometry.PointCloud], list[ClusterSummary]]:
    """Cluster points with DBSCAN, ignore noise, and remove tiny clusters."""
    if pcd.is_empty():
        raise ValueError("Cannot cluster an empty point cloud.")

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
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

    return labels, clusters, summaries


def select_largest_cluster(
    clusters: list[o3d.geometry.PointCloud],
) -> tuple[int, o3d.geometry.PointCloud]:
    """Pick the largest cluster by number of points."""
    if not clusters:
        raise RuntimeError("No object clusters were detected after filtering.")

    sizes = [len(cluster.points) for cluster in clusters]
    selected_index = int(np.argmax(sizes))
    return selected_index, clusters[selected_index]
