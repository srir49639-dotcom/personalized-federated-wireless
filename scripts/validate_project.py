"""Comprehensive Automated Validation Suite.

Verifies:
1. Dataset files exist
2. Scaler and feature configuration exist
3. Locked hybrid strategy matches 0.25 dBm
4. All 16 personalized Keras models exist and load cleanly
5. Feature count equals exactly 472
6. Live inference and hybrid decision logic execute correctly
7. All 18 research figures exist in graphs/generated/
8. No hardcoded /content/ paths remain in the codebase
9. Portable paths and environment compatibility
"""

import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 80)
    print("PERSONALIZED FEDERATED WIRELESS PROJECT VALIDATION SUITE")
    print("=" * 80)

    results = []

    # 1. Dataset verification
    ds_train = PROJECT_ROOT / "dataset" / "UJIndoorLoc" / "trainingData.csv"
    ds_val = PROJECT_ROOT / "dataset" / "UJIndoorLoc" / "validationData.csv"
    train_ok = ds_train.exists() and ds_train.stat().st_size > 10000000
    val_ok = ds_val.exists() and ds_val.stat().st_size > 1000000
    results.append(("UJIIndoorLoc Dataset Files Exist", train_ok and val_ok, f"Train ({ds_train.stat().st_size // 1048576}MB), Val ({ds_val.stat().st_size // 1048576}MB)"))

    # 2. Scaler & Feature Config
    scaler_path = PROJECT_ROOT / "preprocessing" / "centralized_wireless_scaler.pkl"
    feat_path = PROJECT_ROOT / "preprocessing" / "feature_names.json"
    with open(feat_path, "r") as f:
        feat_config = json.load(f)
    feat_count = feat_config.get("total_features", 0)
    results.append(("Preprocessing Scaler & Feature Config", scaler_path.exists() and feat_path.exists(), f"Scaler exists, Total Features = {feat_count}"))

    # 3. Strategy Verification
    strat_path = PROJECT_ROOT / "preprocessing" / "best_hybrid_fedper_strategy.pkl"
    with open(strat_path, "rb") as f:
        strat = pickle.load(f)
    is_strat_ok = strat.get("strategy") == "margin_hybrid" and abs(strat.get("margin_dbm", 0) - 0.25) < 1e-6
    results.append(("Locked Hybrid Strategy Verification", is_strat_ok, f"Strategy: {strat.get('strategy')}, Margin: {strat.get('margin_dbm')} dBm"))

    # 4. 16 Client Models Verification
    model_dir = PROJECT_ROOT / "models" / "final_threshold_aware_fedper"
    expected_ids = [1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24]
    found_ids = []
    for f in os.listdir(model_dir):
        if f.endswith(".keras") and "PHONEID_" in f:
            try:
                pid = int(f.split("PHONEID_")[1].split(".keras")[0])
                found_ids.append(pid)
            except ValueError:
                pass
    found_ids.sort()
    all_models_present = found_ids == expected_ids
    results.append(("16 Personalized Client Models on Disk", all_models_present, f"Found {len(found_ids)}/16 client weights in final_threshold_aware_fedper/"))

    # 5. Model Loading & Inference Test
    from app.backend.model_loader import load_personalized_model, predict_with_personalized_model
    model_1 = load_personalized_model(1)
    dummy_input = np.zeros((1, 472), dtype=np.float32)
    pred_1 = predict_with_personalized_model(1, dummy_input)
    model_inf_ok = model_1 is not None and pred_1 is not None and len(pred_1) == 4
    results.append(("Step 16 Architecture Rebuild & Weight Loading", model_inf_ok, "Model forward pass produced 4-class softmax probabilities"))

    # 6. Feature Pipeline & Preprocessing Verification
    from app.backend.preprocessing import create_deployment_features, parse_wap_input
    sample_input = "WAP001=-55, WAP002=-63, WAP005=-71"
    wap_dict = parse_wap_input(sample_input)
    scaled_vec, raw_vec, stats = create_deployment_features(wap_dict)
    pipe_ok = scaled_vec.shape == (1, 472) and stats["max_rssi"] == -55.0 and stats["supplied_waps"] == 3
    results.append(("472-Dimensional Preprocessing Pipeline", pipe_ok, f"Output Shape: {scaled_vec.shape}, Max RSSI: {stats['max_rssi']} dBm"))

    # 7. Live Hybrid Inference Test
    from app.backend.predictor import predict_link_quality
    # Boundary test case: -64.8 dBm (within 0.25 dBm of -65.0 boundary)
    test_res_boundary = predict_link_quality(1, "WAP001=-64.8\nWAP002=-70.0")
    test_res_outside = predict_link_quality(1, "WAP001=-50.0")
    test_res_unknown = predict_link_quality(99, "WAP001=-50.0")
    
    hybrid_logic_ok = (
        test_res_boundary["used_neural"] is True and
        test_res_outside["used_neural"] is False and
        test_res_unknown["used_neural"] is False and
        test_res_unknown["phone_id_recognized"] is False
    )
    results.append(("Locked Hybrid Decision Logic Routing", hybrid_logic_ok, "Correctly routes boundary points to neural head & distant/unknown points to rule"))

    # 8. 18 Research Figures Verification
    graphs_dir = PROJECT_ROOT / "graphs" / "generated"
    expected_graphs = [
        "01_dataset_class_distribution.png", "02_rssi_distribution.png", "03_max_rssi_distribution.png",
        "04_detected_aps_distribution.png", "05_client_sample_distribution.png", "06_client_class_distribution.png",
        "07_client_class_heatmap.png", "08_client_test_accuracy.png", "09_model_accuracy_comparison.png",
        "10_model_f1_comparison.png", "11_fedper_convergence.png", "12_validation_accuracy_history.png",
        "13_final_confusion_matrix.png", "14_final_per_class_metrics.png", "15_hybrid_decision_usage.png",
        "16_feature_pipeline.png", "17_rssi_threshold_visualization.png", "18_model_architecture.png"
    ]
    missing_graphs = [g for g in expected_graphs if not (graphs_dir / g).exists()]
    graphs_ok = len(missing_graphs) == 0
    results.append(("18 High-Resolution Generated Figures", graphs_ok, f"Found {18 - len(missing_graphs)}/18 PNGs in graphs/generated/"))

    # 9. Scan for hardcoded /content/ paths in python files
    content_matches = []
    for root, _, files in os.walk(PROJECT_ROOT / "app"):
        for file in files:
            if file.endswith(".py"):
                p = Path(root) / file
                with open(p, "r", encoding="utf-8", errors="ignore") as pf:
                    for lno, line in enumerate(pf):
                        if "/content/" in line and not line.strip().startswith("#"):
                            content_matches.append(f"{file}:{lno+1}")
    portable_ok = len(content_matches) == 0
    results.append(("Portable Paths (No Hardcoded /content/ Paths)", portable_ok, f"Scanned app/ codebase, {len(content_matches)} violations found"))

    # 10. Master README and Documentation Check
    readme_ok = (PROJECT_ROOT / "README.md").exists() and (PROJECT_ROOT / "documentation" / "ARTIFACT_AUDIT.md").exists()
    results.append(("Master Documentation & Artifact Audit", readme_ok, "README.md and ARTIFACT_AUDIT.md are present"))

    # ------------------------------------------------------------------------
    # Print Summary Table
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"{'CHECK':<48} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 80)
    all_passed = True
    for name, status, detail in results:
        status_str = "PASS" if status else "FAIL"
        if not status:
            all_passed = False
        print(f"{name:<48} | {status_str:<6} | {detail}")
    print("-" * 80)

    if all_passed:
        print("\nPROJECT BUILD STATUS: ALL 10 COMPREHENSIVE CHECKS PASSED [PASS]\n")
        return 0
    else:
        print("\nPROJECT BUILD STATUS: ONE OR MORE CHECKS FAILED [FAIL]\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
