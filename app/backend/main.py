"""FastAPI Backend Application for Personalized Federated Wireless Link Quality Prediction.

Provides production REST APIs for:
- Live hybrid inference (locked 0.25 dBm boundary margin)
- Artifact-driven benchmarks and research comparisons
- Non-IID dataset analytics and heatmaps
- Training convergence histories
- Static dashboard serving
"""

import io
import json
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.backend.analytics import (
    get_client_metrics,
    get_confusion_matrix,
    get_dataset_samples,
    get_hybrid_strategy_summary,
    get_kpis,
    get_model_comparison,
    get_non_iid_analysis,
    get_training_history,
)
from app.backend.model_loader import (
    VALID_PHONE_IDS,
    _MODEL_CACHE,
    get_project_root,
    load_all_personalized_models,
)
from app.backend.predictor import predict_link_quality
from app.backend.schemas import HealthResponse, PredictionRequest, PredictionResponse

PROJECT_ROOT = get_project_root()
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

# Initialize FastAPI application
app = FastAPI(
    title="Personalized Federated Wireless Link Quality Intelligence",
    description=(
        "Production inference service and research dashboard for RSSI-derived wireless "
        "link quality classification under non-IID cell distributions."
    ),
    version="2.0.0",
)

# CORS middleware for open local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Mount Static Directories (Dashboard & Generated Graphs)
# ----------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if GRAPHS_DIR.exists():
    app.mount("/graphs", StaticFiles(directory=str(GRAPHS_DIR)), name="graphs")

@app.on_event("startup")
async def startup_event():
    """Warm up personalized models in a background thread for instant server readiness."""
    import threading
    def _warmup():
        try:
            from app.backend.model_loader import load_personalized_model
            for pid in [1, 3, 14]:
                load_personalized_model(pid)
        except Exception as e:
            print(f"Warmup notice: {e}")
    threading.Thread(target=_warmup, daemon=True).start()


# ----------------------------------------------------------------------------
# Core Dashboard Route
# ----------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the single-page application dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({
        "status": "online",
        "message": "Dashboard frontend HTML not found in app/frontend/index.html. API endpoints are operational.",
    })

# ----------------------------------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """Health status and model cache inventory."""
    return {
        "status": "healthy",
        "project": "Personalized Federated Wireless Link Quality Prediction",
        "version": "2.0.0",
        "models_cached": len(_MODEL_CACHE),
        "total_clients": len(VALID_PHONE_IDS),
    }

@app.get("/api/project-info", tags=["System"])
async def get_project_info():
    """Metadata regarding dataset, task, and architecture specification."""
    return {
        "project_title": "Personalized Federated Deep Learning for Wireless Link Quality Prediction under Non-IID Cell Data",
        "dataset_name": "UJIIndoorLoc Wi-Fi RSSI Fingerprint Dataset",
        "dataset_clarification": (
            "This system evaluates RSSI-derived wireless link quality using Wi-Fi fingerprint measurements "
            "from the UJIIndoorLoc dataset. It is not a direct 4G/5G cellular dataset."
        ),
        "classes": ["Poor", "Fair", "Good", "Excellent"],
        "num_features": 472,
        "usable_waps": 465,
        "constant_waps_removed": 55,
        "engineered_features": ["max_rssi", "mean_rssi", "median_rssi", "std_rssi", "detected_aps", "top3_mean", "top5_mean"],
        "num_clients": 16,
        "client_phone_ids": VALID_PHONE_IDS,
        "hybrid_strategy": "margin_hybrid",
        "hybrid_margin_dbm": 0.25,
    }

@app.get("/api/metrics", tags=["Analytics"])
async def api_get_kpis():
    """Top-level benchmark KPIs directly from the final held-out test evaluation."""
    return get_kpis()

@app.get("/api/model-comparison", tags=["Analytics"])
async def api_get_model_comparison():
    """Comparative benchmarks across Centralized MLP, FedAvg, FedPer, and Hybrid."""
    return get_model_comparison()

@app.get("/api/client-metrics", tags=["Analytics"])
async def api_get_client_metrics():
    """Per-client test accuracy, sample counts, and status for all 16 PHONEIDs."""
    return get_client_metrics()

@app.get("/api/confusion-matrix", tags=["Analytics"])
async def api_get_confusion_matrix():
    """Exact test confusion matrix, row percentages, and per-class metrics."""
    return get_confusion_matrix()

@app.get("/api/non-iid-analysis", tags=["Analytics"])
async def api_get_non_iid_analysis():
    """Non-IID distributions: class counts and proportions for each client device."""
    return get_non_iid_analysis()

@app.get("/api/training-history", tags=["Analytics"])
async def api_get_training_history():
    """Training loss, accuracy, and round history curves for all models."""
    return get_training_history()

@app.get("/api/hybrid-strategy", tags=["Analytics"])
async def api_get_hybrid_strategy():
    """Locked hybrid strategy parameters and empirical decision distribution."""
    return get_hybrid_strategy_summary()

@app.get("/api/dataset-samples", tags=["Dataset"])
async def api_get_dataset_samples(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    phone_id: Optional[int] = Query(default=None),
):
    """Paginated records from UJIIndoorLoc training dataset."""
    return get_dataset_samples(page=page, page_size=page_size, phone_id=phone_id)

# ----------------------------------------------------------------------------
# Live Prediction Endpoint
# ----------------------------------------------------------------------------

@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def api_predict(request: PredictionRequest):
    """Perform link quality prediction using the locked hybrid system.
    
    Accepts PHONEID and WAP RSSI measurements (dict, comma-separated, or multiline).
    Executes 472-feature pipeline, checks 0.25 dBm boundary margin, and applies
    personalized FedPer neural inference or physical RSSI rule fallback.
    """
    try:
        result = predict_link_quality(
            phone_id=request.phone_id,
            raw_rssi_input=request.rssi,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# ----------------------------------------------------------------------------
# Report & Data Export Endpoints
# ----------------------------------------------------------------------------

@app.get("/api/reports/download/{report_type}", tags=["Reports"])
async def download_report(report_type: str):
    """Download research results and metrics as CSV or JSON."""
    if report_type == "final_predictions":
        file_path = RESULTS_DIR / "FINAL_HYBRID_TEST_PREDICTIONS.csv"
        if file_path.exists():
            return FileResponse(
                str(file_path),
                media_type="text/csv",
                filename="FINAL_HYBRID_TEST_PREDICTIONS.csv",
            )
    elif report_type == "client_metrics":
        clients = get_client_metrics()
        df = pd.DataFrame(clients)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=client_metrics.csv"},
        )
    elif report_type == "model_comparison":
        comp = get_model_comparison()
        df = pd.DataFrame(comp)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=model_comparison.csv"},
        )
    elif report_type == "summary_json":
        summary = {
            "kpis": get_kpis(),
            "model_comparison": get_model_comparison(),
            "client_metrics": get_client_metrics(),
            "confusion_matrix": get_confusion_matrix(),
            "hybrid_strategy": get_hybrid_strategy_summary(),
        }
        return JSONResponse(
            content=summary,
            headers={"Content-Disposition": "attachment; filename=project_summary.json"},
        )
    else:
        raise HTTPException(status_code=404, detail=f"Unknown report type: {report_type}")
