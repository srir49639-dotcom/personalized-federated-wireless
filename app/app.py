
# ============================================================================
# PERSONALIZED FEDERATED DEEP LEARNING
# WIRELESS LINK QUALITY PREDICTION
#
# Standalone deployment application
#
# This file was generated from the tested deployment predictor and
# Gradio UI in Untitled5.ipynb.
# ============================================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ============================================================================
# STEP 19: FINAL DEPLOYABLE HYBRID PREDICTOR
# ============================================================================
#
# FINAL LOCKED STRATEGY:
#   margin_hybrid
#   margin = 0.25 dBm
#
# FINAL TEST PERFORMANCE:
#   Accuracy          = 97.5984%
#   Balanced Accuracy = 98.1312%
#   Weighted F1       = 97.6104%
#
# IMPORTANT:
#   No test data is loaded in this deployment cell.
# ============================================================================

import os
import re
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

print("=" * 80)
print("STEP 19: FINAL DEPLOYABLE HYBRID PREDICTOR")
print("=" * 80)

# ============================================================================
# 1. Paths
# ============================================================================

ROOT_DIR = BASE_DIR if (BASE_DIR / "preprocessing").exists() else BASE_DIR.parent

DATA_PATH = str(ROOT_DIR / "preprocessing" / "federated_data_3way_final.pkl")
SCALER_PATH = str(ROOT_DIR / "preprocessing" / "centralized_wireless_scaler.pkl")

if (ROOT_DIR / "models" / "final_threshold_aware_fedper").exists():
    MODEL_DIR = str(ROOT_DIR / "models" / "final_threshold_aware_fedper")
elif (ROOT_DIR / "models" / "threshold_aware_fedper").exists():
    MODEL_DIR = str(ROOT_DIR / "models" / "threshold_aware_fedper")
else:
    MODEL_DIR = str(ROOT_DIR / "models" / "threshold_aware_fedper")

STRATEGY_PATH = str(ROOT_DIR / "preprocessing" / "best_hybrid_fedper_strategy.pkl")

# ============================================================================
# 2. Load preprocessing metadata ONLY
# ============================================================================

with open(DATA_PATH, "rb") as f:
    deployment_data = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    deployment_scaler = pickle.load(f)

with open(STRATEGY_PATH, "rb") as f:
    deployment_strategy = pickle.load(f)

selected_features = list(
    deployment_data["selected_features"]
)

signal_feature_names = list(
    deployment_data["signal_feature_names"]
)

print("\nDeployment metadata loaded.")

print(
    "WAP features     :",
    len(selected_features)
)

print(
    "Signal features  :",
    len(signal_feature_names)
)

print(
    "Total features   :",
    len(selected_features) + len(signal_feature_names)
)

# ============================================================================
# 3. Locked strategy verification
# ============================================================================

LOCKED_MARGIN = float(
    deployment_strategy["margin_dbm"]
)

if deployment_strategy["strategy"] != "margin_hybrid":
    raise ValueError(
        "Deployment strategy is not margin_hybrid."
    )

if abs(LOCKED_MARGIN - 0.25) > 1e-9:
    raise ValueError(
        f"Expected 0.25 dBm margin, "
        f"found {LOCKED_MARGIN}."
    )

print("\nLocked strategy:")
print(
    "  Strategy :",
    deployment_strategy["strategy"]
)

print(
    "  Margin   :",
    LOCKED_MARGIN,
    "dBm"
)

# ============================================================================
# 4. Feature positions
# ============================================================================

NUM_WAP = len(
    selected_features
)

MAX_RSSI_INDEX = (
    NUM_WAP
    +
    signal_feature_names.index(
        "max_rssi"
    )
)

if MAX_RSSI_INDEX != 465:
    raise ValueError(
        f"Expected max_rssi at position 465, "
        f"found {MAX_RSSI_INDEX}."
    )

# ============================================================================
# 5. Class names
# ============================================================================

CLASS_NAMES = {
    0: "Poor",
    1: "Fair",
    2: "Good",
    3: "Excellent"
}

# ============================================================================
# 6. RSSI thresholds
# ============================================================================

THRESHOLDS = np.array(
    [-75.0, -65.0, -55.0],
    dtype=np.float32
)

# ============================================================================
# 7. Rebuild EXACT Step 16 model architecture
# ============================================================================

scaled_thresholds = (
    THRESHOLDS
    * deployment_scaler.scale_[MAX_RSSI_INDEX]
    +
    deployment_scaler.min_[MAX_RSSI_INDEX]
)

def build_step16_deployment_model():

    inputs = layers.Input(
        shape=(472,),
        name="input"
    )

    # ------------------------------------------------------------------------
    # Shared representation
    # ------------------------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu",
        name="shared_dense_1"
    )(inputs)

    x = layers.LayerNormalization(
        name="shared_ln_1"
    )(x)

    x = layers.Dropout(
        0.08,
        name="shared_dropout_1"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="shared_dense_2"
    )(x)

    x = layers.LayerNormalization(
        name="shared_ln_2"
    )(x)

    x = layers.Dropout(
        0.08,
        name="shared_dropout_2"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="shared_dense_3"
    )(x)

    x = layers.LayerNormalization(
        name="shared_ln_3"
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
        name="shared_dense_4"
    )(x)

    # ------------------------------------------------------------------------
    # max_rssi branch
    # ------------------------------------------------------------------------

    max_rssi = layers.Lambda(
        lambda t:
        t[:, MAX_RSSI_INDEX:MAX_RSSI_INDEX + 1],
        output_shape=(1,),
        name="max_rssi_extract"
    )(inputs)

    rssi_branch = layers.Dense(
        16,
        activation="relu",
        name="rssi_branch_dense"
    )(max_rssi)

    rssi_branch = layers.LayerNormalization(
        name="rssi_branch_ln"
    )(rssi_branch)

    # ------------------------------------------------------------------------
    # Soft threshold branch
    # ------------------------------------------------------------------------

    t75 = float(
        scaled_thresholds[0]
    )

    t65 = float(
        scaled_thresholds[1]
    )

    t55 = float(
        scaled_thresholds[2]
    )

    threshold_features = layers.Lambda(
        lambda t:
        tf.concat(
            [
                tf.sigmoid(
                    40.0 * (t - t75)
                ),
                tf.sigmoid(
                    40.0 * (t - t65)
                ),
                tf.sigmoid(
                    40.0 * (t - t55)
                )
            ],
            axis=1
        ),
        output_shape=(3,),
        name="soft_rssi_thresholds"
    )(max_rssi)

    # ------------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------------

    combined = layers.Concatenate(
        name="shared_feature_fusion"
    )(
        [
            x,
            rssi_branch,
            threshold_features
        ]
    )

    # ------------------------------------------------------------------------
    # Personalized private head
    # ------------------------------------------------------------------------

    private = layers.Dense(
        128,
        activation="relu",
        name="private_dense"
    )(combined)

    private = layers.Dropout(
        0.04,
        name="private_dropout"
    )(private)

    private = layers.Dense(
        64,
        activation="relu",
        name="private_dense_2"
    )(private)

    outputs = layers.Dense(
        4,
        activation="softmax",
        name="classifier"
    )(private)

    return Model(
        inputs,
        outputs,
        name="ThresholdAwareFedPer"
    )

# ============================================================================
# 8. Load personalized client weights
# ============================================================================

client_models = {}

model_files = sorted([
    filename
    for filename in os.listdir(MODEL_DIR)
    if filename.endswith(".keras")
])

if len(model_files) == 0:
    raise FileNotFoundError(
        f"No personalized models found in {MODEL_DIR}"
    )

print(
    "\nLoading personalized client models..."
)

for filename in model_files:

    match = re.search(
        r"PHONEID_(\d+)",
        filename
    )

    if match is None:
        continue

    phone_id = int(
        match.group(1)
    )

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    model = build_step16_deployment_model()

    model.compile(
        optimizer=Adam(
            learning_rate=0.0003
        ),
        loss="sparse_categorical_crossentropy"
    )

    model.load_weights(
        model_path
    )

    client_models[
        phone_id
    ] = model

    print(
        f"  PHONEID {phone_id}: READY"
    )

print(
    "\nPersonalized models loaded:",
    len(client_models)
)

# ============================================================================
# 9. RSSI rule
# ============================================================================

def rssi_rule(
    max_rssi
):

    if max_rssi < -75.0:
        return 0

    elif max_rssi < -65.0:
        return 1

    elif max_rssi < -55.0:
        return 2

    else:
        return 3

# ============================================================================
# 10. Feature construction
# ============================================================================

def create_deployment_features(
    rssi_values
):
    """
    Convert a dictionary of WAP RSSI measurements into the exact
    472-feature representation used by the trained system.
    """

    # ------------------------------------------------------------------------
    # Create 465 WAP vector
    # ------------------------------------------------------------------------

    wap_vector = []

    for feature in selected_features:

        value = rssi_values.get(
            feature,
            -100.0
        )

        try:
            value = float(value)
        except:
            value = -100.0

        # UJIIndoorLoc convention:
        # 100 means no detected signal.
        if value == 100:
            value = -100.0

        wap_vector.append(
            value
        )

    wap_vector = np.asarray(
        wap_vector,
        dtype=np.float32
    )

    # ------------------------------------------------------------------------
    # Signal statistics
    # ------------------------------------------------------------------------

    max_rssi = float(
        np.max(wap_vector)
    )

    mean_rssi = float(
        np.mean(wap_vector)
    )

    median_rssi = float(
        np.median(wap_vector)
    )

    std_rssi = float(
        np.std(wap_vector)
    )

    detected_values = wap_vector[
        wap_vector > -100
    ]

    detected_aps = int(
        len(detected_values)
    )

    if detected_aps > 0:

        sorted_detected = np.sort(
            detected_values
        )[::-1]

        top3_mean = float(
            np.mean(
                sorted_detected[
                    :min(3, detected_aps)
                ]
            )
        )

        top5_mean = float(
            np.mean(
                sorted_detected[
                    :min(5, detected_aps)
                ]
            )
        )

    else:

        top3_mean = -100.0
        top5_mean = -100.0

    engineered = {
        "max_rssi": max_rssi,
        "mean_rssi": mean_rssi,
        "median_rssi": median_rssi,
        "std_rssi": std_rssi,
        "detected_aps": float(detected_aps),
        "top3_mean": top3_mean,
        "top5_mean": top5_mean
    }

    # ------------------------------------------------------------------------
    # Combine 465 + 7 = 472
    # ------------------------------------------------------------------------

    final_vector = wap_vector.tolist()

    for feature in signal_feature_names:

        final_vector.append(
            engineered[feature]
        )

    X = np.asarray(
        final_vector,
        dtype=np.float32
    ).reshape(1, -1)

    if X.shape[1] != 472:
        raise ValueError(
            f"Expected 472 features, "
            f"created {X.shape[1]}."
        )

    return X, engineered

# ============================================================================
# 11. FINAL PREDICTION FUNCTION
# ============================================================================

def predict_final_link_quality(
    rssi_values,
    phone_id=None
):
    """
    Final locked hybrid predictor.

    Parameters
    ----------
    rssi_values:
        Dictionary such as:
        {
            "WAP001": -55,
            "WAP002": -64,
            "WAP003": -100,
            ...
        }

    phone_id:
        Known federated PHONEID.
        Unknown PHONEID -> RSSI-rule fallback.

    Returns
    -------
    Dictionary with final prediction and diagnostics.
    """

    # ------------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------------

    X_raw, stats = create_deployment_features(
        rssi_values
    )

    max_rssi = stats[
        "max_rssi"
    ]

    # ------------------------------------------------------------------------
    # Base RSSI rule
    # ------------------------------------------------------------------------

    rule_class = rssi_rule(
        max_rssi
    )

    # ------------------------------------------------------------------------
    # Distance to closest class boundary
    # ------------------------------------------------------------------------

    distance = float(
        np.min(
            np.abs(
                THRESHOLDS
                -
                max_rssi
            )
        )
    )

    near_threshold = (
        distance <= LOCKED_MARGIN
    )

    # ------------------------------------------------------------------------
    # Determine model
    # ------------------------------------------------------------------------

    neural_used = False
    neural_prediction = None
    neural_confidence = None
    probabilities = None

    final_class = rule_class

    # ------------------------------------------------------------------------
    # Hybrid decision
    # ------------------------------------------------------------------------

    if near_threshold and phone_id in client_models:

        model = client_models[
            phone_id
        ]

        X_scaled = deployment_scaler.transform(
            X_raw
        )

        probabilities = model.predict(
            X_scaled,
            verbose=0
        )[0]

        neural_prediction = int(
            np.argmax(probabilities)
        )

        neural_confidence = float(
            probabilities[
                neural_prediction
            ]
        )

        final_class = neural_prediction
        neural_used = True

        decision_reason = (
            "Near RSSI threshold -> "
            "personalized FedPer"
        )

        model_used = (
            f"Personalized FedPer "
            f"(PHONEID={phone_id})"
        )

    else:

        if phone_id in client_models:

            model_used = (
                f"RSSI rule; "
                f"PHONEID={phone_id}"
            )

        else:

            model_used = (
                "RSSI rule fallback "
                "(unknown PHONEID)"
            )

        decision_reason = (
            "Outside neural threshold region -> "
            "RSSI rule"
        )

    # ------------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------------

    if neural_used:

        confidence = (
            neural_confidence
        )

    else:

        confidence = 1.0

    # ------------------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------------------

    result = {

        "predicted_class":
            int(final_class),

        "predicted_label":
            CLASS_NAMES[int(final_class)],

        "confidence":
            float(confidence),

        "confidence_percent":
            float(confidence * 100),

        "phone_id":
            phone_id,

        "model_used":
            model_used,

        "decision_reason":
            decision_reason,

        "neural_used":
            neural_used,

        "max_rssi_dbm":
            max_rssi,

        "mean_rssi_dbm":
            stats["mean_rssi"],

        "median_rssi_dbm":
            stats["median_rssi"],

        "std_rssi":
            stats["std_rssi"],

        "detected_aps":
            stats["detected_aps"],

        "top3_mean_dbm":
            stats["top3_mean"],

        "top5_mean_dbm":
            stats["top5_mean"],

        "rss_rule_class":
            int(rule_class),

        "rss_rule_label":
            CLASS_NAMES[int(rule_class)],

        "distance_to_threshold_dbm":
            distance
    }

    if probabilities is not None:

        result["neural_probabilities"] = {
            CLASS_NAMES[i]:
            float(probabilities[i])
            for i in range(4)
        }

    return result

# ============================================================================
# 12. Create example input
# ============================================================================

example_input = {
    feature: -100.0
    for feature in selected_features
}

example_input.update({
    "WAP001": -55.2,
    "WAP002": -58.0,
    "WAP005": -61.0,
    "WAP006": -70.0,
    "WAP007": -74.0
})

# ============================================================================
# 13. Test the deployment predictor
# ============================================================================

example_result = predict_final_link_quality(
    example_input,
    phone_id=1
)

print("\n" + "=" * 80)
print("DEPLOYMENT TEST")
print("=" * 80)

print(
    "PHONEID             :",
    example_result["phone_id"]
)

print(
    "Max RSSI            :",
    f"{example_result['max_rssi_dbm']:.2f} dBm"
)

print(
    "Detected APs        :",
    example_result["detected_aps"]
)

print(
    "RSSI-rule class     :",
    example_result["rss_rule_label"]
)

print(
    "Distance to boundary:",
    f"{example_result['distance_to_threshold_dbm']:.2f} dBm"
)

print(
    "Neural model used   :",
    example_result["neural_used"]
)

print(
    "Model/decision      :",
    example_result["model_used"]
)

print(
    "Final prediction    :",
    example_result["predicted_label"]
)

print(
    "Confidence           :",
    f"{example_result['confidence_percent']:.2f}%"
)

# ============================================================================
# 14. Deployment-ready status
# ============================================================================

print("\n" + "=" * 80)
print("STEP 19 COMPLETED")
print("=" * 80)

print(
    "Final predictor     : READY"
)

print(
    "472-feature input   : VERIFIED"
)

print(
    "Locked margin       :",
    LOCKED_MARGIN,
    "dBm"
)

print(
    "Personalized clients:",
    len(client_models)
)

print(
    "Test data loaded    : NO"
)

print(
    "\nUse predict_final_link_quality(...) for deployment."
)

# ============================================================================
# GRADIO INTERACTIVE WEB UI
# ============================================================================

# ============================================================================
# STEP 20: FINAL INTERACTIVE WEB DEMO
# ============================================================================
#
# Uses the locked deployment predictor from STEP 19.
# Does NOT load or use the test set.
# ============================================================================

import json
import traceback

print("=" * 80)
print("STEP 20: FINAL INTERACTIVE WEB DEMO")
print("=" * 80)

# ============================================================================
# 1. Check Gradio
# ============================================================================

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is not installed. Run: !pip install -q gradio"
    )

print("Gradio:", gr.__version__)

# ============================================================================
# 2. Create parser for RSSI input
# ============================================================================

def parse_rssi_text(text):

    """
    Parse input such as:

        WAP001=-55
        WAP002=-63
        WAP005=-71

    or:

        WAP001=-55, WAP002=-63, WAP005=-71
    """

    rssi_values = {}

    if text is None:
        text = ""

    # Support both newline and comma separated input
    normalized = (
        text
        .replace(",", "\n")
        .replace(";", "\n")
    )

    for line in normalized.splitlines():

        line = line.strip()

        if not line:
            continue

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip().upper()
        value = value.strip()

        # Allow accidental spaces such as "WAP001 = -55"
        if not key.startswith("WAP"):
            continue

        try:
            value = float(value)
        except ValueError:
            continue

        # UJIIndoorLoc convention
        if value == 100:
            value = -100.0

        # Keep values in sensible RSSI range
        value = max(
            -100.0,
            min(0.0, value)
        )

        rssi_values[key] = value

    return rssi_values


# ============================================================================
# 3. UI prediction function
# ============================================================================

def web_predict(
    phone_id,
    rssi_text
):

    try:

        # ---------------------------------------------------------------
        # PHONEID
        # ---------------------------------------------------------------

        phone_id_value = None

        if phone_id is not None:

            phone_id_text = str(
                phone_id
            ).strip()

            if phone_id_text:

                try:
                    phone_id_value = int(
                        phone_id_text
                    )
                except ValueError:

                    return (
                        "❌ Invalid PHONEID",
                        "{}"
                    )

        # ---------------------------------------------------------------
        # Parse RSSI input
        # ---------------------------------------------------------------

        rssi_values = parse_rssi_text(
            rssi_text
        )

        if len(rssi_values) == 0:

            return (
                "❌ No valid WAP RSSI values were provided.",
                "{}"
            )

        # ---------------------------------------------------------------
        # Run final predictor
        # ---------------------------------------------------------------

        result = predict_final_link_quality(
            rssi_values,
            phone_id=phone_id_value
        )

        # ---------------------------------------------------------------
        # Format result
        # ---------------------------------------------------------------

        label = result[
            "predicted_label"
        ]

        confidence = result[
            "confidence_percent"
        ]

        neural_used = result[
            "neural_used"
        ]

        model_used = result[
            "model_used"
        ]

        reason = result[
            "decision_reason"
        ]

        # ---------------------------------------------------------------
        # Human-readable result
        # ---------------------------------------------------------------

        output_text = f"""
WIRELESS LINK QUALITY RESULT
========================================

Prediction        : {label}
Confidence        : {confidence:.2f}%

PHONEID           : {phone_id_value}
WAPs supplied     : {len(rssi_values)}
Detected APs      : {result['detected_aps']}

Maximum RSSI      : {result['max_rssi_dbm']:.2f} dBm
Mean RSSI         : {result['mean_rssi_dbm']:.2f} dBm

RSSI-rule result  : {result['rss_rule_label']}
Threshold distance: {result['distance_to_threshold_dbm']:.2f} dBm

Neural model used : {"YES" if neural_used else "NO"}

Decision          : {reason}

Model             : {model_used}
"""

        # ---------------------------------------------------------------
        # JSON details
        # ---------------------------------------------------------------

        json_result = json.dumps(
            result,
            indent=2
        )

        return (
            output_text.strip(),
            json_result
        )

    except Exception as e:

        error_text = (
            "❌ Prediction failed\n\n"
            f"{str(e)}\n\n"
            f"{traceback.format_exc()}"
        )

        return (
            error_text,
            "{}"
        )


# ============================================================================
# 4. Example input
# ============================================================================

example_text = """WAP001=-55.2
WAP002=-58
WAP005=-61
WAP006=-70
WAP007=-74
"""

# ============================================================================
# 5. Build Gradio interface
# ============================================================================

with gr.Blocks(
    title="Federated Wireless Link Quality Predictor"
) as demo:

    gr.Markdown(
        """
# 📡 Federated Wireless Link Quality Predictor

### Personalized Federated Deep Learning + RSSI Hybrid System

Enter a PHONEID and the detected WAP RSSI values.

**Supported quality classes:** Poor, Fair, Good, Excellent
        """
    )

    with gr.Row():

        phone_input = gr.Textbox(
            label="PHONEID",
            value="1",
            placeholder="Example: 1"
        )

    rssi_input = gr.Textbox(
        label="WAP RSSI Measurements",
        value=example_text,
        lines=12,
        placeholder=(
            "WAP001=-55\n"
            "WAP002=-63\n"
            "WAP005=-70"
        )
    )

    predict_button = gr.Button(
        "🔍 Predict Link Quality",
        variant="primary"
    )

    result_output = gr.Textbox(
        label="Prediction Result",
        lines=18
    )

    json_output = gr.Code(
        label="Detailed Prediction JSON",
        language="json",
        lines=20
    )

    predict_button.click(
        fn=web_predict,
        inputs=[
            phone_input,
            rssi_input
        ],
        outputs=[
            result_output,
            json_output
        ]
    )

    gr.Markdown(
        """
### Input format

Use one WAP per line:

`WAP001=-55`

`WAP002=-63`

`WAP005=-71`

Only detected WAPs need to be entered. Missing WAPs are treated as
`-100 dBm`.

For known PHONEIDs, the system can use the personalized FedPer model
near the locked RSSI decision boundaries. Otherwise it uses the RSSI-rule
fallback.
        """
    )

# ============================================================================
# 6. Launch
# ============================================================================

print("\nStarting Gradio interface...")

demo.launch(
    share=False,
    inbrowser=True,
    debug=False
)


print("\n" + "=" * 80)
print("STEP 20 COMPLETED")
print("=" * 80)
print("Interactive deployment UI launched.")