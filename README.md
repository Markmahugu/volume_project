# volume_project

`volume_project` estimates the volume of an object from a 3D point cloud using Open3D. The pipeline loads a `.ply` or `.pcd` file, removes noise and the ground plane, isolates the main object with clustering, and reports volume using both a bounding-box approximation and a voxel-based approximation.

This workflow is relevant to construction monitoring because stockpiles, excavated material, and site assets are often scanned with LiDAR or photogrammetry. Estimating volume from point clouds helps with progress tracking, material auditing, and quantity verification.

## Setup

1. Create and activate a Python 3 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root:

```bash
python main.py
```

By default, the script processes:

```text
WaterTankAndSand_20mm.ply
```

You can also pass a custom point cloud path:

```bash
python main.py --input path/to/cloud.ply
```

## Expected Output

The program prints a short processing summary, including:

- Number of points at each stage
- Number of detected clusters
- Selected cluster index
- Axis-aligned bounding box volume
- Voxel-based estimated volume

Typical console output looks like:

```text
Loading point cloud from: WaterTankAndSand_20mm.ply
Loaded 412356 points.
Downsampled point cloud to 108245 points.
After noise removal: 103998 points.
Removed ground plane. Remaining object points: 28541
Detected 3 cluster(s).
Selected cluster 0 with 21487 points.

Volume Estimation Results
-------------------------
Axis-aligned bounding box volume: 2.1845 m^3
Voxel-based occupied volume:      1.7360 m^3
Estimated Volume: 1.7360 m^3
```

The visualization windows show the processed point cloud, color-coded clusters, and the selected object with its bounding box.
