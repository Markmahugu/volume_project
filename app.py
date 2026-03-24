from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "web" / "static"
TEMPLATE_ROOT = APP_ROOT / "web" / "templates"

templates = Jinja2Templates(directory=TEMPLATE_ROOT)
app = FastAPI(title="volume_project web")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

_WEB_SERVICE_EXPORTS: dict[str, object] | None = None


def _web_service() -> dict[str, object]:
    global _WEB_SERVICE_EXPORTS
    if _WEB_SERVICE_EXPORTS is None:
        try:
            from src.web_service import (
                analyze_selected_region,
                get_cuboid_stats,
                get_preview_payload,
                list_point_cloud_files,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The point-cloud backend could not start because Open3D failed to load. "
                "On this machine the original error was: "
                f"{exc}"
            ) from exc

        _WEB_SERVICE_EXPORTS = {
            "analyze_selected_region": analyze_selected_region,
            "get_cuboid_stats": get_cuboid_stats,
            "get_preview_payload": get_preview_payload,
            "list_point_cloud_files": list_point_cloud_files,
        }
    return _WEB_SERVICE_EXPORTS


class SelectionBounds(BaseModel):
    min: list[float] = Field(min_length=3, max_length=3)
    max: list[float] = Field(min_length=3, max_length=3)


class CuboidSelection(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    yaw: float = 0.0
    snap_to_ground: bool = False
    ground_z: float | None = None


class AnalysisRequest(BaseModel):
    input_path: str
    picked_points: list[list[float]] = Field(default_factory=list)
    selection_mode: Literal["polygon", "box", "cuboid"] = "polygon"
    selection_bounds: SelectionBounds | None = None
    selection_cuboid: CuboidSelection | None = None
    downsample_voxel: float = 0.02
    volume_voxel: float = 0.02
    dbscan_eps: float = 0.12
    dbscan_min_points: int = 100
    min_cluster_size: int = 200
    plane_threshold: float = 0.02
    height_threshold: float = 0.0
    roi_padding_xy: float = 0.5
    roi_padding_z: float = 0.5
    cluster_index: int | None = None


class CuboidStatsRequest(BaseModel):
    input_path: str
    selection_cuboid: CuboidSelection


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    try:
        files = _web_service()["list_point_cloud_files"]()
        backend_error = None
    except RuntimeError as exc:
        files = []
        backend_error = str(exc)
    return templates.TemplateResponse(request, "index.html", {"files": files, "backend_error": backend_error})


@app.get("/api/files")
def get_files() -> dict[str, list[str]]:
    try:
        list_point_cloud_files = _web_service()["list_point_cloud_files"]
        return {"files": list_point_cloud_files()}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/preview")
def get_preview(input_path: str, voxel_size: float = 0.05) -> dict[str, object]:
    try:
        get_preview_payload = _web_service()["get_preview_payload"]
        return get_preview_payload(input_path=input_path, voxel_size=voxel_size)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cuboid-stats")
def cuboid_stats(request: CuboidStatsRequest) -> dict[str, object]:
    try:
        get_cuboid_stats = _web_service()["get_cuboid_stats"]
        return get_cuboid_stats(
            input_path=request.input_path,
            selection_cuboid=request.selection_cuboid.model_dump(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze(request: AnalysisRequest) -> dict[str, object]:
    try:
        payload = request.model_dump()
        analyze_selected_region = _web_service()["analyze_selected_region"]
        summary = analyze_selected_region(**payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "selected_cluster_index": summary.selected_cluster_index,
        "selected_strategy": summary.selected_strategy,
        "cluster_count": summary.cluster_count,
        "cluster_sizes": summary.cluster_sizes,
        "total_points_loaded": summary.total_points_loaded,
        "roi_points": summary.roi_points,
        "downsampled_points": summary.downsampled_points,
        "denoised_points": summary.denoised_points,
        "ground_points": summary.ground_points,
        "object_points": summary.object_points,
        "selected_points": summary.selected_points,
        "points_removed_by_clustering": summary.points_removed_by_clustering,
        "mean_point_spacing": summary.mean_point_spacing,
        "ground_rmse": summary.ground_rmse,
        "roi_completeness": summary.roi_completeness,
        "voxel_count": summary.voxel_count,
        "method_used": summary.method_used,
        "confidence": summary.confidence,
        "error_estimate_percent": summary.error_estimate_percent,
        "warnings": summary.warnings,
        "plane_model": summary.plane_model,
        "bbox_volume_m3": summary.bbox_volume_m3,
        "volume": summary.final_volume_m3,
        "final_volume": summary.final_volume_m3,
        "voxel_volume_m3": summary.final_volume_m3,
        "final_volume_m3": summary.final_volume_m3,
        "binary_voxel_volume_m3": summary.binary_voxel_volume_m3,
        "weighted_voxel_volume_m3": summary.weighted_voxel_volume_m3,
        "height_map_volume_m3": summary.height_map_volume_m3,
        "mesh_volume_m3": summary.mesh_volume_m3,
        "adaptive_voxel_size": summary.adaptive_voxel_size,
        "empty_voxel_percent": summary.empty_voxel_percent,
        "bbox": {"min": summary.bbox_min, "max": summary.bbox_max},
        "ground_cloud": summary.ground_cloud_payload,
        "filtered_cloud": summary.filtered_cloud_payload,
        "selected_cloud": summary.selected_cloud_payload,
        "voxel_cloud": summary.voxel_cloud_payload,
    }
