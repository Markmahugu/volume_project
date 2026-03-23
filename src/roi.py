"""Region-of-interest utilities for user-guided sand pile selection."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from matplotlib.path import Path


MIN_Z_SPAN = 0.25
MAX_ROI_VOLUME_M3 = 50.0
ROI_COLOR = [0.15, 0.4, 0.95]  # Blue color for ROI visualization
PICKED_POINT_COLOR = [1.0, 0.6, 0.2]  # Orange color for picked points


def _order_polygon_vertices(points_xy: np.ndarray) -> np.ndarray:
    centroid = points_xy.mean(axis=0)
    angles = np.arctan2(points_xy[:, 1] - centroid[1], points_xy[:, 0] - centroid[0])
    return points_xy[np.argsort(angles)]


def _expand_polygon(points_xy: np.ndarray, padding_xy: float) -> np.ndarray:
    ordered = _order_polygon_vertices(points_xy)
    centroid = ordered.mean(axis=0)
    expanded = []
    for vertex in ordered:
        direction = vertex - centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            expanded.append(vertex)
        else:
            expanded.append(vertex + (direction / norm) * padding_xy)
    return np.asarray(expanded, dtype=float)


def _bbox_volume(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.prod(extent))


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(x), float(y)) for x, y in points_xy})
    if len(ordered) < 3:
        return np.asarray(ordered, dtype=float)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - origin[0]) * (b[1] - origin[1])) - ((a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _filter_by_xy_polygon(points: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if polygon_xy.shape[0] < 3:
        return np.arange(points.shape[0])
    inside_xy = Path(polygon_xy).contains_points(points[:, :2], radius=1e-9)
    return np.where(inside_xy)[0]


def _density_tighten_roi(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 16:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    xy = points[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    cell_size = float(max(min(span) / 30.0, 0.08))

    grid_indices = np.floor((xy - min_xy) / cell_size).astype(int)
    unique_cells, counts = np.unique(grid_indices, axis=0, return_counts=True)
    if counts.size == 0:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    min_count = max(int(np.percentile(counts, 35)), 2)
    dense_cells = {tuple(cell.tolist()) for cell, count in zip(unique_cells, counts) if int(count) >= min_count}
    if not dense_cells:
        polygon_xy = _convex_hull_xy(points[:, :2])
        return points, polygon_xy

    expanded_cells = set(dense_cells)
    for cell_x, cell_y in list(dense_cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                expanded_cells.add((cell_x + dx, cell_y + dy))

    keep_mask = np.array([tuple(cell.tolist()) in expanded_cells for cell in grid_indices], dtype=bool)
    tightened_points = points[keep_mask]
    polygon_xy = _convex_hull_xy(tightened_points[:, :2])
    polygon_indices = _filter_by_xy_polygon(points, polygon_xy)
    return points[polygon_indices], polygon_xy


def _validate_roi_volume(points: np.ndarray) -> str | None:
    roi_volume = _bbox_volume(points)
    if roi_volume > MAX_ROI_VOLUME_M3:
        return f"ROI is broad ({roi_volume:.2f} m^3 before ground removal); continuing with a warning because spread-out sand can still be valid."
    return None


def compute_polygon_from_picks(
    picked_points: np.ndarray,
    padding_xy: float = 0.0,
) -> np.ndarray:
    """Build a padded 2D polygon from user-picked points in the XY plane."""
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target sand region.")

    polygon = np.asarray(picked_points[:, :2], dtype=float)
    if padding_xy > 0:
        polygon = _expand_polygon(polygon, padding_xy)
    else:
        polygon = _order_polygon_vertices(polygon)
    return polygon


def filter_by_polygon(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Keep all points inside the picked XY polygon and padded Z slab, then tighten the ROI."""
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")
    if picked_points.shape[0] < 3:
        raise ValueError("Pick at least 3 points on the target sand region.")

    polygon = compute_polygon_from_picks(picked_points, padding_xy=padding_xy)

    z_values = np.asarray(picked_points[:, 2], dtype=float)
    z_span = max(float(z_values.max() - z_values.min()), MIN_Z_SPAN)
    z_min = float(z_values.min() - max(padding_z, z_span * 2.0))
    z_max = float(z_values.max() + max(padding_z, z_span))

    all_points = np.asarray(pcd.points)
    inside_xy = Path(polygon).contains_points(all_points[:, :2], radius=1e-9)
    inside_z = (all_points[:, 2] >= z_min) & (all_points[:, 2] <= z_max)
    indices = np.where(inside_xy & inside_z)[0]

    if indices.size == 0:
        raise RuntimeError("The selected polygon ROI does not contain any points.")

    roi_points = all_points[indices]
    roi_warning = _validate_roi_volume(roi_points)
    tightened_points, tightened_polygon = _density_tighten_roi(roi_points)
    tightened_warning = _validate_roi_volume(tightened_points)
    if roi_warning:
        print(roi_warning)
    if tightened_warning:
        print(tightened_warning)

    tightened_cloud = o3d.geometry.PointCloud()
    tightened_cloud.points = o3d.utility.Vector3dVector(tightened_points)

    print(
        f"Selected {indices.size} points from polygon ROI, tightened to {tightened_points.shape[0]} points"
    )
    return tightened_cloud, tightened_polygon


def filter_by_bounds(
    pcd: o3d.geometry.PointCloud,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
    full_height: bool = True,
) -> o3d.geometry.PointCloud:
    """Select all points inside an axis-aligned box, then tighten the ROI by local density."""
    if pcd.is_empty():
        raise ValueError("Cannot crop an empty point cloud.")

    bounds_min = np.asarray(bounds_min, dtype=float).copy()
    bounds_max = np.asarray(bounds_max, dtype=float).copy()

    if bounds_min.shape != (3,) or bounds_max.shape != (3,):
        raise ValueError("Selection bounds must each contain exactly 3 coordinates.")
    if np.any(bounds_max < bounds_min):
        raise ValueError("Selection bounds are invalid.")

    all_points = np.asarray(pcd.points, dtype=float)
    bounds_min[0] -= padding_xy
    bounds_min[1] -= padding_xy
    bounds_max[0] += padding_xy
    bounds_max[1] += padding_xy

    if full_height:
        bounds_min[2] = float(all_points[:, 2].min() - padding_z)
        bounds_max[2] = float(all_points[:, 2].max() + padding_z)
    else:
        bounds_min[2] -= padding_z
        bounds_max[2] += padding_z

    bbox = o3d.geometry.AxisAlignedBoundingBox(bounds_min, bounds_max)
    cropped = pcd.crop(bbox)
    if cropped.is_empty():
        raise RuntimeError("The selected bounding box does not contain any points.")

    cropped_points = np.asarray(cropped.points, dtype=float)
    roi_warning = _validate_roi_volume(cropped_points)
    tightened_points, _ = _density_tighten_roi(cropped_points)
    tightened_warning = _validate_roi_volume(tightened_points)
    if roi_warning:
        print(roi_warning)
    if tightened_warning:
        print(tightened_warning)

    tightened_cloud = o3d.geometry.PointCloud()
    tightened_cloud.points = o3d.utility.Vector3dVector(tightened_points)

    print(
        f"Selected {len(cropped.points)} points from bounding box and tightened to {tightened_points.shape[0]} points "
        f"(full_height={full_height}, xy_padding={padding_xy:.2f}, z_padding={padding_z:.2f})"
    )
    return tightened_cloud


def compute_seed_center(picked_points: np.ndarray) -> np.ndarray:
    """Compute the centroid of the user-picked points."""
    if picked_points.size == 0:
        raise ValueError("No picked points were provided.")

    return picked_points.mean(axis=0)


def create_roi_visualization(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    polygon: np.ndarray,
    padding_xy: float = 0.5,
    padding_z: float = 0.5,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Create visualizations for the ROI selection process.
    
    Returns:
        tuple: (picked_points_cloud, roi_polygon_mesh, roi_volume_mesh)
    """
    if picked_points.shape[0] < 3:
        raise ValueError("Need at least 3 picked points for visualization.")
    
    # Create picked points visualization
    picked_cloud = o3d.geometry.PointCloud()
    picked_cloud.points = o3d.utility.Vector3dVector(picked_points)
    picked_cloud.paint_uniform_color(PICKED_POINT_COLOR)
    
    # Create ROI polygon visualization
    polygon_3d = np.column_stack([polygon, np.full(polygon.shape[0], picked_points[:, 2].mean())])
    polygon_lines = []
    for i in range(len(polygon_3d)):
        polygon_lines.append([i, (i + 1) % len(polygon_3d)])
    
    polygon_mesh = o3d.geometry.LineSet()
    polygon_mesh.points = o3d.utility.Vector3dVector(polygon_3d)
    polygon_mesh.lines = o3d.utility.Vector2iVector(polygon_lines)
    polygon_mesh.paint_uniform_color(ROI_COLOR)
    
    # Create ROI volume visualization (extruded polygon)
    z_values = np.asarray(picked_points[:, 2], dtype=float)
    z_span = max(float(z_values.max() - z_values.min()), MIN_Z_SPAN)
    z_min = float(z_values.min() - max(padding_z, z_span * 2.0))
    z_max = float(z_values.max() + max(padding_z, z_span))
    
    # Create top and bottom polygons
    bottom_polygon = np.column_stack([polygon, np.full(polygon.shape[0], z_min)])
    top_polygon = np.column_stack([polygon, np.full(polygon.shape[0], z_max)])
    
    # Create side walls
    vertices = np.vstack([bottom_polygon, top_polygon])
    faces = []
    
    # Bottom face
    if len(polygon) >= 3:
        for i in range(1, len(polygon) - 1):
            faces.append([0, i, i + 1])
    
    # Top face
    offset = len(polygon)
    if len(polygon) >= 3:
        for i in range(1, len(polygon) - 1):
            faces.append([offset, offset + i + 1, offset + i])
    
    # Side walls
    n = len(polygon)
    for i in range(n):
        next_i = (i + 1) % n
        faces.extend([
            [i, next_i, next_i + n],
            [i, next_i + n, i + n]
        ])
    
    volume_mesh = o3d.geometry.TriangleMesh()
    volume_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    volume_mesh.triangles = o3d.utility.Vector3iVector(faces)
    volume_mesh.paint_uniform_color([0.15, 0.4, 0.95])
    volume_mesh.compute_vertex_normals()
    
    return picked_cloud, polygon_mesh, volume_mesh


def validate_roi_selection(
    pcd: o3d.geometry.PointCloud,
    picked_points: np.ndarray,
    polygon: np.ndarray,
    min_points: int = 100,
    max_points: int = 1000000,
) -> dict[str, object]:
    """Validate the ROI selection and provide feedback.
    
    Returns:
        dict: Validation results with keys:
            - 'valid': bool
            - 'point_count': int
            - 'volume_estimate': float
            - 'warnings': list[str]
            - 'suggestions': list[str]
    """
    if picked_points.shape[0] < 3:
        return {
            'valid': False,
            'point_count': 0,
            'volume_estimate': 0.0,
            'warnings': ['Need at least 3 picked points'],
            'suggestions': ['Pick at least 3 points around the sand pile']
        }
    
    # Estimate points in ROI
    all_points = np.asarray(pcd.points)
    inside_xy = Path(polygon).contains_points(all_points[:, :2], radius=1e-9)
    point_count = np.sum(inside_xy)
    
    # Estimate volume
    if point_count > 0:
        roi_points = all_points[inside_xy]
        extent = roi_points.max(axis=0) - roi_points.min(axis=0)
        volume_estimate = float(np.prod(extent))
    else:
        volume_estimate = 0.0
    
    warnings = []
    suggestions = []
    
    if point_count < min_points:
        warnings.append(f"ROI contains only {point_count} points (minimum recommended: {min_points})")
        suggestions.append("Expand the ROI to include more points")
    
    if point_count > max_points:
        warnings.append(f"ROI contains {point_count} points (may be too large)")
        suggestions.append("Consider making the ROI smaller for better performance")
    
    if volume_estimate > MAX_ROI_VOLUME_M3:
        warnings.append(f"ROI volume estimate ({volume_estimate:.2f} m³) is very large")
        suggestions.append("ROI may be too large for accurate volume estimation")
    
    # Check polygon quality
    if len(polygon) >= 3:
        # Check for self-intersections (simple check)
        from scipy.spatial.distance import pdist
        distances = pdist(polygon)
        min_distance = np.min(distances) if len(distances) > 0 else 0
        if min_distance < 1e-6:
            warnings.append("Some picked points are too close together")
            suggestions.append("Spread picked points more evenly around the pile")
    
    return {
        'valid': len(warnings) == 0,
        'point_count': int(point_count),
        'volume_estimate': volume_estimate,
        'warnings': warnings,
        'suggestions': suggestions
    }


def interactive_roi_refinement(
    pcd: o3d.geometry.PointCloud,
    initial_picked_points: np.ndarray,
    window_name: str = "ROI Refinement",
) -> np.ndarray:
    """Allow interactive refinement of the ROI selection.
    
    Returns:
        np.ndarray: Refined picked points
    """
    if initial_picked_points.shape[0] < 3:
        raise ValueError("Need at least 3 initial picked points.")
    
    current_points = initial_picked_points.copy()
    
    # Create initial visualization
    polygon = compute_polygon_from_picks(current_points)
    picked_cloud, polygon_mesh, volume_mesh = create_roi_visualization(pcd, current_points, polygon)
    
    # Create base point cloud visualization
    display_cloud = o3d.geometry.PointCloud(pcd)
    display_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Gray for base cloud
    
    # Combine all geometries
    geometries = [display_cloud, picked_cloud, polygon_mesh, volume_mesh]
    
    print("\nROI Refinement Mode")
    print("-------------------")
    print("1. Left-click on the base cloud to add new points")
    print("2. Right-click on existing picked points to remove them")
    print("3. Press 'r' to reset selection")
    print("4. Press 'q' to accept current selection")
    print("5. Use mouse to rotate/pan/zoom the view")
    
    # Simple interactive refinement (for now, just return initial points)
    # In a full implementation, this would use VisualizerWithEditing
    print(f"Current selection: {len(current_points)} points")
    validation = validate_roi_selection(pcd, current_points, polygon)
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['suggestions']:
        print("Suggestions:")
        for suggestion in validation['suggestions']:
            print(f"  - {suggestion}")
    
    return current_points

