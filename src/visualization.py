"""Visualization utilities for processed point clouds."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


BACKGROUND_COLOR = np.array([0.08, 0.08, 0.1])


def _draw_geometries_with_camera(
    geometries: list[o3d.geometry.Geometry],
    reference_geometry: o3d.geometry.Geometry,
    window_name: str,
    point_size: float = 2.0,
) -> None:
    """Render geometry with an explicit camera fit to avoid blank initial views."""
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=window_name, width=1280, height=720)

    for geometry in geometries:
        visualizer.add_geometry(geometry)

    render_option = visualizer.get_render_option()
    render_option.background_color = BACKGROUND_COLOR
    render_option.point_size = point_size

    view_control = visualizer.get_view_control()
    bounds = reference_geometry.get_axis_aligned_bounding_box()
    view_control.set_lookat(bounds.get_center())

    extent = np.asarray(bounds.get_extent(), dtype=float)
    largest_extent = float(np.max(extent))
    zoom = 0.7 if largest_extent > 5.0 else 0.9
    view_control.set_front([0.8, -0.4, -0.45])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(zoom)

    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.run()
    visualizer.destroy_window()


def show_point_cloud(pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud") -> None:
    """Display a point cloud in an Open3D viewer."""
    if pcd.is_empty():
        raise ValueError("Cannot visualize an empty point cloud.")

    display_cloud = o3d.geometry.PointCloud(pcd)
    if not display_cloud.has_colors():
        display_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    _draw_geometries_with_camera([display_cloud], display_cloud, window_name)


def show_clusters(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    window_name: str = "Clustered Objects",
) -> None:
    """Color each DBSCAN cluster label for inspection."""
    if pcd.is_empty():
        raise ValueError("Cannot visualize clusters for an empty point cloud.")

    colored_cloud = o3d.geometry.PointCloud(pcd)
    max_label = int(labels.max()) if labels.size > 0 else -1

    if max_label < 0:
        colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(labels), 1))
    else:
        colormap = plt.get_cmap("tab20")
        colors = np.zeros((len(labels), 3), dtype=float)
        for index, label in enumerate(labels):
            if label < 0:
                colors[index] = [0.25, 0.25, 0.25]
            else:
                colors[index] = colormap((label % 20) / 19 if 19 > 0 else 0)[:3]

    colored_cloud.colors = o3d.utility.Vector3dVector(colors)
    _draw_geometries_with_camera([colored_cloud], colored_cloud, window_name)


def show_with_bbox(
    pcd: o3d.geometry.PointCloud,
    bbox: o3d.geometry.AxisAlignedBoundingBox,
    window_name: str = "Selected Object with Bounding Box",
) -> None:
    """Display the selected point cloud together with its bounding box."""
    if pcd.is_empty():
        raise ValueError("Cannot visualize an empty point cloud.")

    display_cloud = o3d.geometry.PointCloud(pcd)
    display_cloud.paint_uniform_color([0.1, 0.7, 0.2])
    bbox.color = (1.0, 0.0, 0.0)
    _draw_geometries_with_camera([display_cloud, bbox], display_cloud, window_name, point_size=3.0)
