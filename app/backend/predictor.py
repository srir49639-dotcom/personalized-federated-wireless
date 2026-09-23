"""Locked Hybrid Link Quality Predictor.

Implements the locked production strategy (margin_hybrid, margin = 0.25 dBm):
- Fast and reliable RSSI physical boundary rule for points outside the margin
- High-precision personalized FedPer neural classification within the 0.25 dBm margin
- Graceful deterministic fallback for unrecognized client PHONEIDs
"""

from typing import Any, Dict, Optional, Union
import numpy as np

from app.backend.model_loader import (
    VALID_PHONE_IDS,
    predict_with_personalized_model,
)
from app.backend.preprocessing import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    compute_threshold_distance,
    create_deployment_features,
    parse_wap_input,
    rssi_rule_prediction,
)

LOCKED_STRATEGY = "margin_hybrid"
LOCKED_MARGIN_DBM = 0.25

def predict_link_quality(
    phone_id: Optional[Union[int, str]],
    raw_rssi_input: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Perform hybrid link quality prediction.
    
    Args:
        phone_id: Client identifier (integer 1-24, or None / "unknown")
        raw_rssi_input: Dictionary of WAPs or raw text string
        
    Returns:
        Structured prediction result dictionary.
    """
    # 1. Parse and validate phone ID
    client_id: Optional[int] = None
    if phone_id is not None:
        try:
            client_id = int(phone_id)
        except (ValueError, TypeError):
            client_id = None

    # 2. Parse WAP input
    wap_dict = parse_wap_input(raw_rssi_input)

    # 3. Create 472-feature vector and compute signal stats
    scaled_vector, raw_vector, summary_stats = create_deployment_features(wap_dict)
    max_rssi = summary_stats["max_rssi"]

    # 4. Compute RSSI Rule classification and boundary distance
    rule_class = rssi_rule_prediction(max_rssi)
    nearest_threshold, distance_to_threshold = compute_threshold_distance(max_rssi)

    # 5. Execute locked hybrid decision logic
    is_known_client = client_id is not None and client_id in VALID_PHONE_IDS
    within_margin = distance_to_threshold <= LOCKED_MARGIN_DBM

    used_neural = False
    model_used = "RSSI Threshold Rule"
    neural_probs: Optional[list] = None
    neural_class: Optional[int] = None

    if is_known_client and within_margin:
        # Client recognized and near decision boundary: invoke personalized FedPer model
        probs = predict_with_personalized_model(client_id, scaled_vector)
        if probs is not None:
            used_neural = True
            neural_probs = [float(p) for p in probs]
            neural_class = int(np.argmax(probs))
            final_class = neural_class
            confidence = float(probs[final_class])
            model_used = f"Personalized FedPer (PHONEID {client_id})"
            decision_reason = (
                f"Personalized neural model selected because the maximum RSSI ({max_rssi:.2f} dBm) "
                f"is within the locked hybrid boundary margin ({distance_to_threshold:.2f} dBm <= {LOCKED_MARGIN_DBM} dBm) "
                f"of threshold {nearest_threshold:.1f} dBm."
            )
        else:
            final_class = rule_class
            confidence = 1.0
            model_used = "RSSI Threshold Rule (Model Load Fallback)"
            decision_reason = "Model weights unavailable — fallback RSSI rule applied."
    elif not is_known_client:
        final_class = rule_class
        confidence = 1.0
        model_used = "RSSI Threshold Rule (Unregistered Client Fallback)"
        decision_reason = (
            f"Unregistered / Unknown PHONEID ({phone_id}) — fallback RSSI physical decision rule used."
        )
    else:
        final_class = rule_class
        confidence = 1.0
        model_used = "RSSI Threshold Rule"
        decision_reason = (
            f"RSSI-rule decision — maximum RSSI ({max_rssi:.2f} dBm) is outside the locked hybrid margin "
            f"(distance: {distance_to_threshold:.2f} dBm > {LOCKED_MARGIN_DBM} dBm)."
        )

    class_name = CLASS_NAMES[final_class]
    rule_class_name = CLASS_NAMES[rule_class]

    # Format class probabilities
    probabilities = {}
    if neural_probs is not None:
        for idx, prob in enumerate(neural_probs):
            probabilities[CLASS_NAMES[idx]] = round(prob, 4)
    else:
        for idx in range(4):
            probabilities[CLASS_NAMES[idx]] = 1.0 if idx == final_class else 0.0

    return {
        "predicted_class": final_class,
        "class_name": class_name,
        "class_description": CLASS_DESCRIPTIONS[final_class],
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "phone_id": client_id,
        "phone_id_recognized": is_known_client,
        "used_neural": used_neural,
        "model_used": model_used,
        "decision_reason": decision_reason,
        "strategy": LOCKED_STRATEGY,
        "hybrid_margin_dbm": LOCKED_MARGIN_DBM,
        "rule_class": rule_class,
        "rule_class_name": rule_class_name,
        "neural_class": neural_class,
        "nearest_threshold_dbm": nearest_threshold,
        "distance_to_threshold_dbm": round(distance_to_threshold, 3),
        "within_boundary_margin": within_margin,
        "signal_stats": summary_stats,
    }
