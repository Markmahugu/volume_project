# volume_project

`volume_project` is a browser-based and CLI-driven sand volume estimation tool built on Open3D. The updated pipeline uses a user-defined ROI polygon, stronger ground removal, height filtering, DBSCAN cluster filtering, and voxel occupancy so the final volume estimate is driven by the sand pile itself instead of nearby site clutter.

## Setup

1. Create and activate a Python 3 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

Start the backend from the project root:

```bash
python -m uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Run the CLI

```bash
python main.py --input WaterTankAndSand_20mm.ply
```

The CLI prompts for ROI picks in Open3D, then prints stage counts and the final voxel-based volume.

## Why The ROI Matters

Sand piles are often scanned together with tanks, curbs, machines, walls, and bare ground. The ROI step converts the user-picked points into a padded 2D polygon in the XY plane and keeps only points inside that footprint. That removes most unrelated geometry before any clustering happens, which makes the downstream segmentation much more stable.

## Why Height Filtering Improves Accuracy

RANSAC plane fitting removes the dominant ground plane, but rough sites often leave residual low points near the floor. Height filtering removes points within a small offset above the local minimum Z after ground removal. This strips out thin ground remnants and helps DBSCAN focus on the pile body instead of edge debris.

## Why Voxel Volume Is More Reliable

The axis-aligned bounding box is useful as a comparison value, but it tends to overestimate because it includes empty space around irregular pile shapes. The voxel method counts only occupied cells, so it tracks the real stockpile geometry more closely. In this pipeline, `Voxel Volume (FINAL)` is the primary estimate and the bounding-box result is reported only for reference.

## Updated Pipeline

1. Load point cloud.
2. Build a padded ROI polygon from the user-picked points.
3. Filter the cloud to the ROI footprint.
4. Downsample and denoise.
5. Remove the dominant ground plane.
6. Apply height filtering to remove residual near-ground points.
7. Run DBSCAN and discard tiny clusters.
8. Select the largest remaining cluster by default.
9. Compute the final voxel volume and a comparison bounding-box volume.

## Visualization Colors

- Ground: green
- Filtered object cloud: red
- Selected pile: blue
- Bounding box: amber
- Occupied voxels: cyan

## Key Files

- [main.py](E:/volume_project/main.py): robust CLI pipeline with stage-by-stage diagnostics
- [app.py](E:/volume_project/app.py): FastAPI entry point and API routes
- [src/roi.py](E:/volume_project/src/roi.py): polygon ROI construction and filtering
- [src/filters.py](E:/volume_project/src/filters.py): height filtering utilities
- [src/segmentation.py](E:/volume_project/src/segmentation.py): stronger ground-plane removal
- [src/clustering.py](E:/volume_project/src/clustering.py): DBSCAN clustering and small-cluster filtering
- [src/volume.py](E:/volume_project/src/volume.py): voxel-first volume estimation helpers
- [src/web_service.py](E:/volume_project/src/web_service.py): web pipeline backend
- [web/static/app.js](E:/volume_project/web/static/app.js): canvas viewer and analysis overlay rendering
