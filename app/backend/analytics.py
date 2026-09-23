"""Analytics and Reporting Module.

Aggregates and formats real metrics, confusion matrices, non-IID distributions,
client benchmarks, and training histories directly from repository artifacts.
No synthetic metrics or fabricated results.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "app" / "frontend" / "index.html").exists():
            return parent
        if (parent / "results" / "FINAL_HYBRID_TEST_METRICS.pkl").exists():
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
DATASET_DIR = PROJECT_ROOT / "dataset" / "UJIndoorLoc"

# ----------------------------------------------------------------------------
# In-Memory Precomputed Cache
# ----------------------------------------------------------------------------
_ANALYTICS_CACHE: Dict[str, Any] = {}

def get_kpis() -> Dict[str, Any]:
    """Return top-level KPI metrics from actual stored test evaluations."""
    if "kpis" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["kpis"]

    metrics_path = RESULTS_DIR / "FINAL_HYBRID_TEST_METRICS.pkl"
    with open(metrics_path, "rb") as f:
        hm = pickle.load(f)

    # Convert np.float64 to native float for JSON serialization
    kpis = {
        "dataset_name": "UJIIndoorLoc Wi-Fi Fingerprint",
        "dataset_samples": 19937,
        "external_samples": 1111,
        "total_samples": 21048,
        "federated_clients": 16,
        "model_features": 472,
        "wap_features": 465,
        "engineered_features": 7,
        "hybrid_test_accuracy": round(float(hm["hybrid_test_accuracy"]) * 100, 2),
        "hybrid_balanced_accuracy": round(float(hm["hybrid_balanced_accuracy"]) * 100, 2),
        "hybrid_weighted_f1": round(float(hm["hybrid_weighted_f1"]) * 100, 2),
        "hybrid_weighted_precision": round(float(hm["hybrid_weighted_precision"]) * 100, 2),
        "hybrid_weighted_recall": round(float(hm["hybrid_weighted_recall"]) * 100, 2),
        "neural_usage_percentage": round(float(hm["neural_usage_fraction"]) * 100, 2),
        "test_samples": int(hm["test_samples"]),
        "locked_strategy": hm["strategy"],
        "locked_margin_dbm": float(hm["margin_dbm"]),
    }
    _ANALYTICS_CACHE["kpis"] = kpis
    return kpis

def get_model_comparison() -> List[Dict[str, Any]]:
    """Return comparative metrics across Centralized, FedAvg, FedPer, and Hybrid."""
    if "model_comparison" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["model_comparison"]

    comp_path = RESULTS_DIR / "FINAL_RESULTS_COMPARISON.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
    else:
        df = pd.DataFrame()

    kpis = get_kpis()
    comparison = []
    for _, row in df.iterrows():
        comparison.append({
            "model": row["Model"],
            "accuracy": round(float(row["Test Accuracy"]) * 100, 2),
            "balanced_accuracy": round(float(row["Balanced Accuracy"]) * 100, 2),
            "weighted_f1": round(float(row["Weighted F1"]) * 100, 2),
            "is_hybrid": False,
        })

    # Add final locked hybrid system
    comparison.append({
        "model": "Locked Hybrid FedPer (Production)",
        "accuracy": kpis["hybrid_test_accuracy"],
        "balanced_accuracy": kpis["hybrid_balanced_accuracy"],
        "weighted_f1": kpis["hybrid_weighted_f1"],
        "is_hybrid": True,
    })

    _ANALYTICS_CACHE["model_comparison"] = comparison
    return comparison

def get_client_metrics() -> List[Dict[str, Any]]:
    """Return per-client sample counts and test accuracy across all 16 PHONEIDs."""
    if "client_metrics" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["client_metrics"]

    client_res_path = RESULTS_DIR / "final_fedper_client_test_results.csv"
    df_res = pd.read_csv(client_res_path)

    # Load split sample counts from federated_data_3way_final.pkl
    fed_path = PREPROCESSING_DIR / "federated_data_3way_final.pkl"
    with open(fed_path, "rb") as f:
        d = pickle.load(f)

    train_counts = pd.Series(d["clients_train"]).value_counts().to_dict()
    val_counts = pd.Series(d["clients_val"]).value_counts().to_dict()
    test_counts = pd.Series(d["clients_test"]).value_counts().to_dict()

    clients = []
    for _, row in df_res.iterrows():
        client_idx = int(row["Client"])
        pid = int(row["PHONEID"])
        tr = train_counts.get(client_idx, 0)
        va = val_counts.get(client_idx, 0)
        te = test_counts.get(client_idx, int(row["Test_Samples"]))
        total = tr + va + te

        clients.append({
            "client_index": client_idx,
            "phone_id": pid,
            "train_samples": tr,
            "val_samples": va,
            "test_samples": te,
            "total_samples": total,
            "fedper_test_accuracy": round(float(row["Accuracy"]) * 100, 2),
            "fedper_balanced_accuracy": round(float(row["Balanced_Accuracy"]) * 100, 2),
            "status": "Ready (Inference Model Cached)",
        })

    _ANALYTICS_CACHE["client_metrics"] = clients
    return clients

def get_confusion_matrix() -> Dict[str, Any]:
    """Return the final hybrid confusion matrix and per-class precision/recall/F1."""
    if "confusion_matrix" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["confusion_matrix"]

    pred_path = RESULTS_DIR / "FINAL_HYBRID_TEST_PREDICTIONS.csv"
    df = pd.read_csv(pred_path)

    class_names = ["Poor", "Fair", "Good", "Excellent"]
    cm = confusion_matrix(df["actual_class"], df["hybrid_prediction"], labels=[0, 1, 2, 3])
    cm_list = cm.tolist()

    # Compute row percentages (normalized by true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.where(row_sums > 0, (cm / row_sums) * 100, 0)
    cm_percent_list = [[round(float(val), 1) for val in row] for row in cm_percent]

    # Classification report
    cr = classification_report(
        df["actual_class"],
        df["hybrid_prediction"],
        target_names=class_names,
        output_dict=True,
    )

    per_class = []
    for c in class_names:
        per_class.append({
            "class_name": c,
            "precision": round(cr[c]["precision"] * 100, 2),
            "recall": round(cr[c]["recall"] * 100, 2),
            "f1_score": round(cr[c]["f1-score"] * 100, 2),
            "support": int(cr[c]["support"]),
        })

    result = {
        "class_names": class_names,
        "matrix_counts": cm_list,
        "matrix_percentages": cm_percent_list,
        "per_class": per_class,
        "overall": {
            "accuracy": round(cr["accuracy"] * 100, 2),
            "macro_f1": round(cr["macro avg"]["f1-score"] * 100, 2),
            "weighted_f1": round(cr["weighted avg"]["f1-score"] * 100, 2),
            "total_test_samples": len(df),
        },
    }
    _ANALYTICS_CACHE["confusion_matrix"] = result
    return result

def get_non_iid_analysis() -> Dict[str, Any]:
    """Compute real non-IID distributions across the 16 clients from actual federated data."""
    if "non_iid" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["non_iid"]

    fed_path = PREPROCESSING_DIR / "federated_data_3way_final.pkl"
    with open(fed_path, "rb") as f:
        d = pickle.load(f)

    unique_clients = d["unique_clients"]  # [1, 3, 6, ...]
    client_indices = np.concatenate([d["clients_train"], d["clients_val"], d["clients_test"]])
    labels = np.concatenate([d["y_train"], d["y_val"], d["y_test"]])

    class_names = ["Poor", "Fair", "Good", "Excellent"]
    client_class_matrix = []
    client_proportions = []

    for c_idx, pid in enumerate(unique_clients):
        c_mask = np.array(client_indices) == c_idx
        c_labels = labels[c_mask]
        total_c = len(c_labels)
        counts = [int(np.sum(c_labels == k)) for k in range(4)]
        props = [round((cnt / total_c) * 100, 1) if total_c > 0 else 0.0 for cnt in counts]

        client_class_matrix.append({
            "phone_id": pid,
            "client_index": c_idx,
            "total_samples": total_c,
            "counts": counts,
        })
        client_proportions.append({
            "phone_id": pid,
            "client_index": c_idx,
            "percentages": props,
        })

    result = {
        "class_names": class_names,
        "phone_ids": unique_clients,
        "counts_matrix": client_class_matrix,
        "proportions_matrix": client_proportions,
        "client_sample_sizes": [
            {"phone_id": item["phone_id"], "samples": item["total_samples"]}
            for item in client_class_matrix
        ],
    }
    _ANALYTICS_CACHE["non_iid"] = result
    return result

def get_training_history() -> Dict[str, Any]:
    """Return epoch/round progression for Centralized, FedAvg, and FedPer models."""
    if "training_history" in _ANALYTICS_CACHE:
        return _ANALYTICS_CACHE["training_history"]

    # 1. Centralized history
    cent_path = RESULTS_DIR / "centralized_training_history.csv"
    cent_data = []
    if cent_path.exists():
        df_c = pd.read_csv(cent_path)
        for ep, row in df_c.iterrows():
            cent_data.append({
                "epoch": ep + 1,
                "loss": round(float(row["loss"]), 4),
                "val_loss": round(float(row["val_loss"]), 4),
                "accuracy": round(float(row["accuracy"]) * 100, 2),
                "val_accuracy": round(float(row["val_accuracy"]) * 100, 2),
            })

    # 2. FedAvg round history
    fedavg_path = RESULTS_DIR / "fedavg_round_history.csv"
    fedavg_data = []
    if fedavg_path.exists():
        df_fa = pd.read_csv(fedavg_path)
        for _, row in df_fa.iterrows():
            fedavg_data.append({
                "round": int(row["round"]),
                "train_acc": round(float(row["mean_client_train_accuracy"]) * 100, 2),
                "val_acc": round(float(row["validation_accuracy"]) * 100, 2),
                "val_balanced_acc": round(float(row["validation_balanced_accuracy"]) * 100, 2),
                "val_f1": round(float(row["validation_weighted_f1"]) * 100, 2),
            })

    # 3. Final FedPer round history
    fedper_path = RESULTS_DIR / "final_fedper_round_history.csv"
    fedper_data = []
    if fedper_path.exists():
        df_fp = pd.read_csv(fedper_path)
        for _, row in df_fp.iterrows():
            fedper_data.append({
                "round": int(row["round"]),
                "train_acc": round(float(row["train_accuracy"]) * 100, 2),
                "mean_client_val": round(float(row["mean_client_validation"]) * 100, 2),
                "pooled_val": round(float(row["pooled_validation"]) * 100, 2),
                "balanced_val": round(float(row["balanced_validation"]) * 100, 2),
                "weighted_f1": round(float(row["weighted_f1_validation"]) * 100, 2),
            })

    # 4. Threshold-aware FedPer validation history
    thresh_path = RESULTS_DIR / "threshold_aware_fedper_validation_history.csv"
    thresh_data = []
    if thresh_path.exists():
        df_th = pd.read_csv(thresh_path)
        for _, row in df_th.iterrows():
            thresh_data.append({
                "round": int(row["round"]),
                "mean_client_acc": round(float(row["mean_client_accuracy"]) * 100, 2),
                "pooled_acc": round(float(row["pooled_accuracy"]) * 100, 2),
                "balanced_acc": round(float(row["balanced_accuracy"]) * 100, 2),
                "weighted_f1": round(float(row["weighted_f1"]) * 100, 2),
            })

    history = {
        "centralized": cent_data,
        "fedavg": fedavg_data,
        "fedper_final": fedper_data,
        "threshold_aware_fedper": thresh_data,
    }
    _ANALYTICS_CACHE["training_history"] = history
    return history

def get_hybrid_strategy_summary() -> Dict[str, Any]:
    """Return details on locked hybrid strategy and empirical test set usage."""
    pred_path = RESULTS_DIR / "FINAL_HYBRID_TEST_PREDICTIONS.csv"
    df = pd.read_csv(pred_path)

    used_neural_count = int(df["used_neural"].sum())
    used_rule_count = int(len(df) - used_neural_count)
    total_count = int(len(df))

    return {
        "strategy": "margin_hybrid",
        "margin_dbm": 0.25,
        "thresholds": [-75.0, -65.0, -55.0],
        "total_test_samples": total_count,
        "neural_decisions_count": used_neural_count,
        "rule_decisions_count": used_rule_count,
        "neural_usage_percentage": round((used_neural_count / total_count) * 100, 2),
        "rule_usage_percentage": round((used_rule_count / total_count) * 100, 2),
        "rationale": (
            "Points clearly inside a signal quality regime (> 0.25 dBm from boundary) are classified "
            "with zero latency via the physical RSSI threshold rule. Points in the ambiguity zone "
            "within 0.25 dBm of the boundary are classified by the client's personalized neural head."
        ),
    }

def get_dataset_samples(page: int = 1, page_size: int = 10, phone_id: Optional[int] = None) -> Dict[str, Any]:
    """Return paginated rows from trainingData.csv for the dataset explorer."""
    train_csv = DATASET_DIR / "trainingData.csv"
    if not train_csv.exists():
        return {"total_rows": 0, "page": page, "page_size": page_size, "rows": []}

    # Read selected columns to keep response small and fast
    meta_cols = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP", "LATITUDE", "LONGITUDE", "FLOOR", "BUILDINGID"]
    
    # Read CSV
    df = pd.read_csv(train_csv)
    if phone_id is not None:
        df = df[df["PHONEID"] == phone_id]

    total_rows = len(df)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    sub_df = df.iloc[start_idx:end_idx].copy()
    
    # Extract top 5 active WAPs for display
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    rows = []
    for _, r in sub_df.iterrows():
        active_waps = {}
        for w in wap_cols:
            val = r[w]
            if val != 100:
                active_waps[w] = float(val)
        
        # Max RSSI
        max_rssi = max(active_waps.values()) if active_waps else -100.0
        
        rows.append({
            "phone_id": int(r["PHONEID"]),
            "building_id": int(r["BUILDINGID"]),
            "floor": int(r["FLOOR"]),
            "detected_waps_count": len(active_waps),
            "max_rssi_dbm": round(max_rssi, 1),
            "top_detected_waps": dict(sorted(active_waps.items(), key=lambda item: item[1], reverse=True)[:5]),
            "timestamp": int(r["TIMESTAMP"]),
        })

    return {
        "total_rows": total_rows,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_rows + page_size - 1) // page_size,
        "rows": rows,
    }
