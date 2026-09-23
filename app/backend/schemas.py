"""Pydantic Schemas for API Requests and Responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    phone_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Client device identifier (e.g. 1, 3, 14 or 'unknown')",
    )
    rssi: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Detected WAP RSSI measurements in dict, multi-line string, or comma-separated string",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "phone_id": 1,
                "rssi": {
                    "WAP001": -55.0,
                    "WAP002": -63.0,
                    "WAP005": -71.0,
                    "WAP010": -68.0,
                },
            }
        }

class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    class_description: str
    confidence: float
    probabilities: Dict[str, float]
    phone_id: Optional[int]
    phone_id_recognized: bool
    used_neural: bool
    model_used: str
    decision_reason: str
    strategy: str
    hybrid_margin_dbm: float
    rule_class: int
    rule_class_name: str
    neural_class: Optional[int]
    nearest_threshold_dbm: float
    distance_to_threshold_dbm: float
    within_boundary_margin: bool
    signal_stats: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    models_cached: int
    total_clients: int
