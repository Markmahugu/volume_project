"""Point cloud clustering helpers."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def cluster_objects(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.05,
    min_points: int = 30,
) -> tuple[np.ndarray, list[o3d.geometry.PointCloud]]:
    """Cluster points with DBSCAN and return labels plus extracted clusters."""
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
    clusters = [pcd.select_by_index(np.where(labels == label)[0]) for label in valid_labels]

    return labels, clusters
