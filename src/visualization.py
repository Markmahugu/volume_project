"""Visualization utilities for processed point clouds."""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


BACKGROUND_COLOR = np.array([0.08, 0.08, 0.1])
MAX_RENDER_POINTS = 250000


def _downsample_for_display(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    display_cloud = o3d.geometry.PointCloud(pcd)
    point_count = len(display_cloud.points)
    if point_count > MAX_RENDER_POINTS:
        sample_ratio = MAX_RENDER_POINTS / point_count
        display_cloud = display_cloud.random_down_sample(sample_ratio)
    return display_cloud


def _apply_uniform_color(
    pcd: o3d.geometry.PointCloud,
    color: tuple[float, float, float] | None = None,
) -> o3d.geometry.PointCloud:
    colored = o3d.geometry.PointCloud(pcd)
    if color is not None:
        colored.paint_uniform_color(color)
    elif not colored.has_colors():
        colored.paint_uniform_color([0.85, 0.85, 0.85])
    return colored


def _configure_visualizer(
    visualizer: o3d.visualization.Visualizer,
    largest_extent: float,
    point_size: float,
) -> None:
    render_option = visualizer.get_render_option()
    render_option.background_color = BACKGROUND_COLOR
    render_option.point_size = point_size
    render_option.light_on = True

    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.reset_view_point(True)

    view_control = visualizer.get_view_control()
    view_control.set_lookat([0.0, 0.0, 0.0])
    view_control.set_front([0.9, -0.35, -0.25])
    view_control.set_up([0.0, 0.0, 1.0])
    if largest_extent > 50:
        view_control.set_zoom(0.25)
    elif largest_extent > 10:
        view_control.set_zoom(0.4)
    else:
        view_control.set_zoom(0.7)

    visualizer.poll_events()
    visualizer.update_renderer()


def _center_geometries(
    geometries: list[o3d.geometry.Geometry],
    bbox_source: o3d.geometry.PointCloud,
) -> tuple[list[o3d.geometry.Geometry], float]:
    bbox = bbox_source.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=float)
    largest_extent = max(float(np.max(bbox.get_extent())), 1.0)

    centered: list[o3d.geometry.Geometry] = []
    for geometry in geometries:
        clone = copy.deepcopy(geometry)
        clone.translate(-center)
        centered.append(clone)
    return centered, largest_extent


def _draw_geometries_with_camera(
    geometries: list[o3d.geometry.Geometry],
    bbox_source: o3d.geometry.PointCloud,
    window_name: str,
    point_size: float = 3.0,
) -> None:
    centered_geometries, largest_extent = _center_geometries(geometries, bbox_source)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=window_name, width=1280, height=720, visible=True)

    for geometry in centered_geometries:
        visualizer.add_geometry(geometry, reset_bounding_box=False)

    _configure_visualizer(visualizer, largest_extent, point_size)
    visualizer.run()
    visualizer.destroy_window()


def pick_points_for_roi(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "Pick Sand Region",
) -> np.ndarray:
    """Let the user pick seed points that define the target sand region."""
    if pcd.is_empty():
        raise ValueError("Cannot pick points from an empty point cloud.")

    display_cloud = _apply_uniform_color(_downsample_for_display(pcd), (0.85, 0.85, 0.85))
    bbox = display_cloud.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=float)
    largest_extent = max(float(np.max(bbox.get_extent())), 1.0)
    display_cloud.translate(-center)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(largest_extent * 0.1, 0.5))

    print("\nInteractive ROI selection")
    print("-------------------------")
    print("1. Hold Shift and left-click to pick points on the sand pile.")
    print("2. Pick at least 3 points spread across the target region.")
    print("3. Press Shift + right-click to undo a pick if needed.")
    print("4. Press Q to finish selection and continue processing.\n")

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name=window_name, width=1280, height=720, visible=True)
    visualizer.add_geometry(display_cloud, reset_bounding_box=False)
    visualizer.add_geometry(frame, reset_bounding_box=False)
    _configure_visualizer(visualizer, largest_extent, point_size=5.0)
    visualizer.run()
    picked_indices = visualizer.get_picked_points()
    visualizer.destroy_window()

    if len(picked_indices) < 3:
        raise RuntimeError("At least 3 picked points are required to build an ROI.")

    return np.asarray(display_cloud.points)[picked_indices] + center


def show_clusters(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    window_name: str = "Clustered Objects",
) -> None:
    if pcd.is_empty():
        raise ValueError("Cannot visualize clusters for an empty point cloud.")
    if len(labels) != len(pcd.points):
        raise ValueError("Label count must match the number of points in the cloud.")

    point_count = len(pcd.points)
    if point_count > MAX_RENDER_POINTS:
        sample_indices = np.random.choice(point_count, MAX_RENDER_POINTS, replace=False)
        sample_indices.sort()
        display_cloud = pcd.select_by_index(sample_indices.tolist())
        display_labels = labels[sample_indices]
    else:
        display_cloud = o3d.geometry.PointCloud(pcd)
        display_labels = labels

    max_label = int(display_labels.max()) if display_labels.size > 0 else -1
    if max_label < 0:
        colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(display_labels), 1))
    else:
        colormap = plt.get_cmap("tab20")
        colors = np.zeros((len(display_labels), 3), dtype=float)
        for index, label in enumerate(display_labels):
            colors[index] = [0.35, 0.35, 0.35] if label < 0 else colormap((label % 20) / 19.0)[:3]

    display_cloud.colors = o3d.utility.Vector3dVector(colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(np.max(display_cloud.get_axis_aligned_bounding_box().get_extent()) * 0.1, 0.5))
    _draw_geometries_with_camera([display_cloud, frame], display_cloud, window_name)


def show_pipeline_result(
    ground_cloud: o3d.geometry.PointCloud,
    filtered_object_cloud: o3d.geometry.PointCloud,
    selected_cluster: o3d.geometry.PointCloud,
    bbox: o3d.geometry.AxisAlignedBoundingBox,
    window_name: str = "Sand Segmentation Result",
) -> None:
    """Show ground, filtered objects, selected pile, and the bbox in one aligned view."""
    if selected_cluster.is_empty():
        raise ValueError("Cannot visualize an empty selected cluster.")

    geometries: list[o3d.geometry.Geometry] = []
    if not ground_cloud.is_empty():
        geometries.append(_apply_uniform_color(_downsample_for_display(ground_cloud), (0.1, 0.75, 0.2)))
    if not filtered_object_cloud.is_empty():
        geometries.append(_apply_uniform_color(_downsample_for_display(filtered_object_cloud), (0.9, 0.2, 0.2)))
    geometries.append(_apply_uniform_color(_downsample_for_display(selected_cluster), (0.15, 0.4, 0.95)))

    display_bbox = copy.deepcopy(bbox)
    display_bbox.color = (1.0, 0.75, 0.1)
    geometries.append(display_bbox)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(float(np.max(selected_cluster.get_axis_aligned_bounding_box().get_extent())) * 0.1, 0.5)
    )
    geometries.append(frame)
    _draw_geometries_with_camera(geometries, selected_cluster, window_name, point_size=4.0)


def show_roi_selection(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    polygon: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
    window_name: str = "ROI Selection",
) -> None:
    """Show the ROI selection with enhanced visual feedback."""
    from src.roi import create_roi_visualization
    
    if picked_points.shape[0] < 3:
        raise ValueError("Need at least 3 picked points for ROI visualization.")
    
    # Create ROI visualizations
    picked_cloud, polygon_mesh, volume_mesh = create_roi_visualization(
        pcd, picked_points, polygon, padding_xy, padding_z
    )
    
    # Create base point cloud visualization (downsampled for performance)
    display_cloud = _apply_uniform_color(_downsample_for_display(pcd), (0.8, 0.8, 0.8))
    
    geometries = [display_cloud, picked_cloud, polygon_mesh, volume_mesh]
    
    print(f"\nROI Selection Preview")
    print(f"---------------------")
    print(f"Picked points: {len(picked_points)}")
    print(f"Polygon vertices: {len(polygon)}")
    print(f"ROI visualization includes:")
    print(f"  - Gray base cloud (downsampled)")
    print(f"  - Orange picked points")
    print(f"  - Blue ROI polygon outline")
    print(f"  - Blue ROI volume mesh")
    
    _draw_geometries_with_camera(geometries, pcd, window_name, point_size=3.0)


def show_roi_validation(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    validation_result: dict,
    window_name: str = "ROI Validation",
) -> None:
    """Show ROI validation results with visual feedback."""
    from src.roi import compute_polygon_from_picks, create_roi_visualization
    
    if picked_points.shape[0] < 3:
        raise ValueError("Need at least 3 picked points for validation.")
    
    polygon = compute_polygon_from_picks(picked_points)
    picked_cloud, polygon_mesh, volume_mesh = create_roi_visualization(pcd, picked_points, polygon)
    
    # Create base point cloud visualization
    display_cloud = _apply_uniform_color(_downsample_for_display(pcd), (0.8, 0.8, 0.8))
    
    geometries = [display_cloud, picked_cloud, polygon_mesh, volume_mesh]
    
    print(f"\nROI Validation Results")
    print(f"----------------------")
    print(f"Selection valid: {'Yes' if validation_result['valid'] else 'No'}")
    print(f"Point count: {validation_result['point_count']:,}")
    print(f"Volume estimate: {validation_result['volume_estimate']:.4f} m³")
    
    if validation_result['warnings']:
        print(f"\nWarnings:")
        for warning in validation_result['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation_result['suggestions']:
        print(f"\nSuggestions:")
        for suggestion in validation_result['suggestions']:
            print(f"  💡 {suggestion}")
    
    _draw_geometries_with_camera(geometries, pcd, window_name, point_size=3.0)
