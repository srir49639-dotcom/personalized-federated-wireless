"""Preprocessing and Feature Engineering Pipeline.

Implements the exact 472-feature pipeline used in the trained system:
1. 465 usable WAP features (55 constant WAPs removed from 520 raw WAPs)
2. 7 engineered RSSI statistical features
3. MinMaxScaler normalization using centralized scaler
4. Exact RSSI decision boundaries and threshold distance calculation
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np

# ----------------------------------------------------------------------------
# 1. Load Preprocessing Metadata
# ----------------------------------------------------------------------------

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "app" / "frontend" / "index.html").exists():
            return parent
        if (parent / "preprocessing" / "feature_names.json").exists() and parent.name != "app":
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = get_project_root()
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

# Load feature names and configuration
with open(PREPROCESSING_DIR / "feature_names.json", "r") as f:
    FEATURE_CONFIG = json.load(f)

SELECTED_FEATURES: List[str] = FEATURE_CONFIG.get(
    "feature_names", []
)[:465]  # First 465 are usable WAP features

SIGNAL_FEATURE_NAMES: List[str] = [
    "max_rssi",
    "mean_rssi",
    "median_rssi",
    "std_rssi",
    "detected_aps",
    "top3_mean",
    "top5_mean",
]

# Load fitted MinMaxScaler
with open(PREPROCESSING_DIR / "centralized_wireless_scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# Class definitions
CLASS_NAMES = {
    0: "Poor",
    1: "Fair",
    2: "Good",
    3: "Excellent",
}

CLASS_DESCRIPTIONS = {
    0: "Poor: max RSSI < -75 dBm (Unstable connection, frequent packet loss)",
    1: "Fair: -75 dBm <= max RSSI < -65 dBm (Adequate for voice/low-bitrate data)",
    2: "Good: -65 dBm <= max RSSI < -55 dBm (Reliable broadband streaming and video)",
    3: "Excellent: max RSSI >= -55 dBm (Optimal throughput, ultra-low latency link)",
}

RSSI_THRESHOLDS = [-75.0, -65.0, -55.0]

# ----------------------------------------------------------------------------
# 2. Input Parser
# ----------------------------------------------------------------------------

def parse_wap_input(raw_input: Union[str, Dict[str, Any]]) -> Dict[str, float]:
    """Parse WAP RSSI measurements from string or dictionary format.
    
    Supports:
    - Dict: {"WAP001": -55.0, "WAP002": -63.0}
    - Multiline string: "WAP001=-55\nWAP002=-63"
    - Comma-separated string: "WAP001=-55, WAP002=-63, WAP005=-71"
    - Colon format: "WAP001: -55, WAP002: -63"
    """
    if isinstance(raw_input, dict):
        parsed = {}
        for k, v in raw_input.items():
            name = str(k).strip().upper()
            try:
                val = float(v)
                # UJIIndoorLoc 100 means no signal
                if val == 100.0:
                    val = -100.0
                parsed[name] = val
            except (ValueError, TypeError):
                continue
        return parsed

    parsed = {}
    if not isinstance(raw_input, str):
        return parsed

    # Match patterns like WAP001=-55 or WAP001: -55
    pattern = re.compile(r"([A-Za-z0-9_]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)")
    for match in pattern.finditer(raw_input):
        wap_name = match.group(1).upper()
        try:
            val = float(match.group(2))
            if val == 100.0:
                val = -100.0
            parsed[wap_name] = val
        except ValueError:
            continue

    return parsed

# ----------------------------------------------------------------------------
# 3. Feature Pipeline Construction (472 dimensions)
# ----------------------------------------------------------------------------

def create_deployment_features(rssi_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Transform raw WAP measurements into the exact 472-feature vector and compute stats.
    
    Returns:
        scaled_vector: (1, 472) normalized vector ready for neural network inference
        raw_vector: (1, 472) unscaled vector with physical dBm values
        summary_stats: Dict of extracted signal characteristics
    """
    # 1. Construct 465 WAP vector
    wap_vector = []
    for feat in SELECTED_FEATURES:
        val = rssi_values.get(feat, -100.0)
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = -100.0

        if val == 100.0:
            val = -100.0
        wap_vector.append(val)

    wap_arr = np.asarray(wap_vector, dtype=np.float32)

    # 2. Compute 7 engineered features
    max_rssi = float(np.max(wap_arr))
    mean_rssi = float(np.mean(wap_arr))
    median_rssi = float(np.median(wap_arr))
    std_rssi = float(np.std(wap_arr))

    detected_values = wap_arr[wap_arr > -100.0]
    detected_aps = int(len(detected_values))

    if detected_aps > 0:
        sorted_detected = np.sort(detected_values)
        top3_mean = float(np.mean(sorted_detected[-3:]))
        top5_mean = float(np.mean(sorted_detected[-5:]))
    else:
        top3_mean = -100.0
        top5_mean = -100.0

    engineered_arr = np.array(
        [
            max_rssi,
            mean_rssi,
            median_rssi,
            std_rssi,
            float(detected_aps),
            top3_mean,
            top5_mean,
        ],
        dtype=np.float32,
    )

    # 3. Combine into exact 472-dimensional vector
    full_vector = np.concatenate([wap_arr, engineered_arr])
    raw_vector = np.expand_dims(full_vector, axis=0)

    # 4. Apply fitted MinMaxScaler
    scaled_vector = SCALER.transform(raw_vector)

    summary_stats = {
        "max_rssi": max_rssi,
        "mean_rssi": round(mean_rssi, 2),
        "median_rssi": round(median_rssi, 2),
        "std_rssi": round(std_rssi, 2),
        "detected_aps": detected_aps,
        "top3_mean": round(top3_mean, 2),
        "top5_mean": round(top5_mean, 2),
        "supplied_waps": len(rssi_values),
    }

    return scaled_vector, raw_vector, summary_stats

# ----------------------------------------------------------------------------
# 4. RSSI Threshold Rule & Boundary Math
# ----------------------------------------------------------------------------

def rssi_rule_prediction(max_rssi: float) -> int:
    """Classify wireless link quality based on exact max RSSI threshold bounds.
    
    Poor: < -75 dBm (class 0)
    Fair: -75 to < -65 dBm (class 1)
    Good: -65 to < -55 dBm (class 2)
    Excellent: >= -55 dBm (class 3)
    """
    if max_rssi < -75.0:
        return 0
    elif max_rssi < -65.0:
        return 1
    elif max_rssi < -55.0:
        return 2
    else:
        return 3

def compute_threshold_distance(max_rssi: float) -> Tuple[float, float]:
    """Calculate the absolute distance to the closest RSSI decision boundary.
    
    Returns:
        (nearest_threshold, min_distance_dbm)
    """
    distances = [abs(max_rssi - t) for t in RSSI_THRESHOLDS]
    min_idx = int(np.argmin(distances))
    return RSSI_THRESHOLDS[min_idx], float(distances[min_idx])
