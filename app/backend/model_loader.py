"""Personalized Federated Model Loader.

Reconstructs the exact Step 16 Threshold-Aware FedPer neural architecture
with explicit Lambda layer shapes to safely deserialize trained weights
across all 16 client models without retraining or architecture drift.
"""

import os
import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# ----------------------------------------------------------------------------
# 1. Project Paths & Metadata Setup
# ----------------------------------------------------------------------------

def get_project_root() -> Path:
    """Find repository root containing app/frontend/index.html or models/ and preprocessing/ directories."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "app" / "frontend" / "index.html").exists():
            return parent
        if (parent / "preprocessing" / "centralized_wireless_scaler.pkl").exists() and (parent / "models").exists() and parent.name != "app":
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = get_project_root()
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
SCALER_PATH = PREPROCESSING_DIR / "centralized_wireless_scaler.pkl"

# Model directory resolution (checks final_threshold_aware_fedper first)
if (PROJECT_ROOT / "models" / "final_threshold_aware_fedper").exists():
    MODEL_DIR = PROJECT_ROOT / "models" / "final_threshold_aware_fedper"
elif (PROJECT_ROOT / "models" / "threshold_aware_fedper").exists():
    MODEL_DIR = PROJECT_ROOT / "models" / "threshold_aware_fedper"
else:
    MODEL_DIR = PROJECT_ROOT / "models" / "final_threshold_aware_fedper"

# Load scaler for scaled RSSI soft threshold parameters
with open(SCALER_PATH, "rb") as f:
    _SCALER = pickle.load(f)

MAX_RSSI_INDEX = 465
RAW_THRESHOLDS = np.array([-75.0, -65.0, -55.0], dtype=np.float32)
SCALED_THRESHOLDS = (
    RAW_THRESHOLDS * _SCALER.scale_[MAX_RSSI_INDEX] + _SCALER.min_[MAX_RSSI_INDEX]
)
T75 = float(SCALED_THRESHOLDS[0])
T65 = float(SCALED_THRESHOLDS[1])
T55 = float(SCALED_THRESHOLDS[2])

VALID_PHONE_IDS = [1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24]

# ----------------------------------------------------------------------------
# 2. Step 16 Threshold-Aware FedPer Architecture Builder
# ----------------------------------------------------------------------------

def build_step16_deployment_model() -> Model:
    """Rebuild the exact Step 16 Threshold-Aware FedPer neural architecture.
    
    Contains:
    - 472-dimensional input
    - Shared representation trunk (Dense + LayerNorm + Dropout)
    - Direct max_rssi extraction branch with LayerNorm
    - Soft continuous RSSI threshold sigmoid branch
    - Shared feature fusion
    - Client-specific personalized private classification head
    """
    inputs = layers.Input(shape=(472,), name="input")

    # Shared representation trunk
    x = layers.Dense(256, activation="relu", name="shared_dense_1")(inputs)
    x = layers.LayerNormalization(name="shared_ln_1")(x)
    x = layers.Dropout(0.08, name="shared_dropout_1")(x)

    x = layers.Dense(256, activation="relu", name="shared_dense_2")(x)
    x = layers.LayerNormalization(name="shared_ln_2")(x)
    x = layers.Dropout(0.08, name="shared_dropout_2")(x)

    x = layers.Dense(128, activation="relu", name="shared_dense_3")(x)
    x = layers.LayerNormalization(name="shared_ln_3")(x)

    x = layers.Dense(64, activation="relu", name="shared_dense_4")(x)

    # max_rssi extract branch
    max_rssi = layers.Lambda(
        lambda t: t[:, MAX_RSSI_INDEX:MAX_RSSI_INDEX + 1],
        output_shape=(1,),
        name="max_rssi_extract"
    )(inputs)

    rssi_branch = layers.Dense(16, activation="relu", name="rssi_branch_dense")(max_rssi)
    rssi_branch = layers.LayerNormalization(name="rssi_branch_ln")(rssi_branch)

    # Continuous soft threshold branch
    threshold_features = layers.Lambda(
        lambda t: tf.concat(
            [
                tf.sigmoid(40.0 * (t - T75)),
                tf.sigmoid(40.0 * (t - T65)),
                tf.sigmoid(40.0 * (t - T55)),
            ],
            axis=1
        ),
        output_shape=(3,),
        name="soft_rssi_thresholds"
    )(max_rssi)

    # Feature fusion
    combined = layers.Concatenate(name="shared_feature_fusion")(
        [x, rssi_branch, threshold_features]
    )

    # Personalized private head
    private = layers.Dense(128, activation="relu", name="private_dense")(combined)
    private = layers.Dropout(0.04, name="private_dropout")(private)
    private = layers.Dense(64, activation="relu", name="private_dense_2")(private)
    outputs = layers.Dense(4, activation="softmax", name="classifier")(private)

    return Model(inputs, outputs, name="ThresholdAwareFedPer")

# ----------------------------------------------------------------------------
# 3. Model Cache & Loading Functions
# ----------------------------------------------------------------------------

_MODEL_CACHE: Dict[int, Model] = {}

def get_model_path_for_client(phone_id: int) -> Optional[Path]:
    """Return file path for a client's .keras model weights."""
    target_name = f"client_PHONEID_{phone_id}.keras"
    direct_path = MODEL_DIR / target_name
    if direct_path.exists():
        return direct_path

    # Fallback to search in MODEL_DIR
    if MODEL_DIR.exists():
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(".keras") and f"PHONEID_{phone_id}." in fname:
                return MODEL_DIR / fname
    return None

def load_personalized_model(phone_id: int) -> Optional[Model]:
    """Load and cache the personalized model for a specific PHONEID.
    
    Reconstructs the model architecture and loads the trained weights.
    Returns None if the phone_id is not in the recognized client list.
    """
    if phone_id in _MODEL_CACHE:
        return _MODEL_CACHE[phone_id]

    if phone_id not in VALID_PHONE_IDS:
        return None

    model_path = get_model_path_for_client(phone_id)
    if model_path is None or not model_path.exists():
        return None

    model = build_step16_deployment_model()
    model.load_weights(str(model_path))
    _MODEL_CACHE[phone_id] = model
    return model

def load_all_personalized_models() -> Dict[int, Model]:
    """Preload and cache all 16 client models."""
    for pid in VALID_PHONE_IDS:
        if pid not in _MODEL_CACHE:
            load_personalized_model(pid)
    return _MODEL_CACHE

def predict_with_personalized_model(phone_id: int, scaled_features: np.ndarray) -> Optional[np.ndarray]:
    """Run inference for a specific client model on scaled 472-dim input vector.
    
    Args:
        phone_id: Client identifier
        scaled_features: NumPy array of shape (1, 472) or (472,)
        
    Returns:
        NumPy array of class probabilities of shape (4,), or None if model unavailable.
    """
    model = load_personalized_model(phone_id)
    if model is None:
        return None

    if scaled_features.ndim == 1:
        scaled_features = np.expand_dims(scaled_features, axis=0)

    probs = model.predict(scaled_features, verbose=0)
    return probs[0]
