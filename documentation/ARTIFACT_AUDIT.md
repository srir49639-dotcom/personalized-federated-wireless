# Project Artifact Audit

**Project Title:** Personalized Federated Deep Learning for Wireless Link Quality Prediction under Non-IID Cell Data  
**Audit Date:** September 2026  
**Auditor:** ML & Systems Engineering Team  
**Dataset Reference:** UJIIndoorLoc Wi-Fi RSSI Fingerprint Dataset (RSSI-derived link quality)

---

## 1. Executive Summary

This document provides a complete audit and classification of all project artifacts, models, preprocessing pipelines, datasets, results, histories, and notebooks within the repository.

The repository contains both:
1. **Production / Final Deployment Artifacts**: The locked hybrid FedPer system with 16 personalized client models, 472-feature pipeline, centralized scaler, and 0.25 dBm boundary margin.
2. **Research & Experimental Artifacts**: Baseline models (Centralized MLP, FedAvg, FedPer candidate architectures A/B/C/D, corrected FedPer models), round-by-round convergence logs, and parameter tuning sweeps.

---

## 2. Artifact Classification Taxonomy

```
Artifacts
├── Final Deployment Artifacts (Required for runtime inference & production UI)
├── Research & Experimental Artifacts (Required for model comparison, baselines, and ablation)
├── Analytics & Dataset Artifacts (Required for reproducible non-IID analysis and feature inspection)
└── Legacy & Development Artifacts (Notebook logs, extracted source material)
```

---

## 3. Final Deployment Artifacts

These artifacts are strictly frozen and utilized by the production prediction service and dashboard.

| Artifact Path | Type | Description | Key Properties / Dimensions |
| :--- | :--- | :--- | :--- |
| `models/final_threshold_aware_fedper/client_PHONEID_*.keras` | Keras Models (16 files) | Personalized client neural models containing shared layers + client private head + RSSI boundary branch | **Clients:** 1, 3, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24<br>**Input:** (472,)<br>**Output:** Softmax over 4 classes |
| `preprocessing/centralized_wireless_scaler.pkl` | Scikit-learn `MinMaxScaler` | Fitted MinMaxScaler on training set for all 472 features | **Shape:** 472 features<br>**Range:** [0, 1] |
| `preprocessing/feature_names.json` | JSON Metadata | Exact list and order of 465 WAPs + 7 engineered features | **Total Features:** 472<br>**WAPs Used:** 465<br>**Constant Removed:** 55 |
| `preprocessing/best_hybrid_fedper_strategy.pkl` | Python Dictionary | Locked production decision strategy | **Strategy:** `margin_hybrid`<br>**Margin:** `0.25 dBm`<br>**Validation Acc:** `97.73%` |
| `results/FINAL_HYBRID_TEST_METRICS.pkl` | Metrics Dict | Final evaluation results on held-out test set (2,998 samples) | **Accuracy:** `97.93%`<br>**Balanced Acc:** `98.42%`<br>**Weighted F1:** `97.94%`<br>**Neural Usage:** `6.50%` |
| `results/FINAL_HYBRID_TEST_PREDICTIONS.csv` | Prediction Table | Instance-level test evaluation table for all 2,998 test samples | **Rows:** 2,998<br>**Columns:** `test_index`, `phone_id`, `max_rssi_dbm`, `rule_prediction`, `neural_prediction`, `hybrid_prediction`, `actual_class`, `used_neural` |
| `results/final_fedper_client_test_results.csv` | Per-client Results | Breakdown of test accuracy and sample count per PHONEID | **Clients:** 16<br>**Mean Accuracy:** `87.41%` (Neural FedPer alone), `97.93%` (Hybrid) |

---

## 4. Research & Experimental Artifacts

These artifacts document the evolutionary research steps, baselines, and parameter tuning sweeps.

| Artifact Path | Component | Description |
| :--- | :--- | :--- |
| `models/centralized/best_centralized_wireless_mlp.keras` | Centralized Baseline | Standard MLP trained on all pooled client data (Test Acc: 93.13%, Weighted F1: 93.13%) |
| `models/fedavg/fedavg_wireless_link_quality_FINAL.keras` | Federated Baseline | Standard FedAvg model without personalization (Test Acc: 82.89%, Weighted F1: 82.65%) |
| `models/research/fedper_candidates/config_*.pkl` | Architecture Search | Candidate FedPer splits (Configurations A, B, C, D) exploring shared vs private layer depths |
| `models/research/fedper_corrected/client_*.keras` | Intermediate Milestone | Corrected FedPer architecture with revised normalization layers |
| `models/research/fedper_final/client_*.keras` | Intermediate Milestone | Step 15 FedPer models before threshold-aware boundary integration |
| `results/centralized_training_history.csv` | Training Log | 36 epochs of centralized model loss and validation accuracy |
| `results/fedavg_round_history.csv` | Federation Log | 20 rounds of FedAvg aggregation and validation metrics |
| `results/fedper_corrected_round_history.csv` | Federation Log | 25 rounds of corrected FedPer federated personalization rounds |
| `results/final_fedper_round_history.csv` | Federation Log | 35 rounds of final FedPer federated training |
| `results/threshold_aware_fedper_validation_history.csv` | Federation Log | 35 rounds of threshold-aware FedPer client training |
| `results/hybrid_fedper_validation_results.csv` | Strategy Tuning | 55-row grid search over margin thresholds (0.0 to 3.0 dBm) and alpha values |
| `results/FINAL_PROJECT_RESULTS.csv` | Research Summary | Benchmark comparison table across Centralized, FedAvg, and FedPer |
| `results/FINAL_RESULTS_COMPARISON.csv` | Research Summary | Standardized table comparing Accuracy, Balanced Accuracy, and Weighted F1 |

---

## 5. Analytics & Dataset Artifacts

| Artifact Path | Format | Description | Records / Details |
| :--- | :--- | :--- | :--- |
| `dataset/UJIndoorLoc/trainingData.csv` | Raw CSV | Primary UJIIndoorLoc Wi-Fi RSSI fingerprint data | **Rows:** 19,937 used (21,048 total)<br>**Columns:** 529 (520 WAPs, Coordinates, Floor, Building, Space, RelativePosition, UserID, PhoneID, Timestamp) |
| `dataset/UJIndoorLoc/validationData.csv` | Raw CSV | External validation / test fingerprint data | **Rows:** 1,111 samples |
| `preprocessing/federated_data_3way_final.pkl` | Pickled Dataset | Partitioned federated splits for the 16 non-IID clients | **Train:** 13,949 samples (70%)<br>**Val:** 2,990 samples (15%)<br>**Test:** 2,998 samples (15%) |
| `graphs/` | PNG Images | High-resolution analytical charts documenting convergence and accuracy | `fedper_client_accuracy.png`, `fedper_convergence.png`, `final_metrics_comparison.png`, `test_accuracy_comparison.png` |

---

## 6. Legacy & Source Material Artifacts

| Artifact Path | Purpose |
| :--- | :--- |
| `notebooks/PERSONALIZED_FEDERATED_WIRELESS_COMPLETE.ipynb` | Complete end-to-end Google Colab experimental notebook (preserved as research provenance) |
| `source_material/experiment_configuration.pkl` | Original Colab runtime environment parameters |
| `source_material/extracted_gradio_ui_cells.txt` | Extracted cell scripts from legacy Gradio prototype |
| `source_material/project_file_inventory.csv` | Colab file snapshot inventory |
| `documentation/PROJECT_METADATA.json` | Project metadata summary |
| `documentation/PACKAGE_MANIFEST.csv` | File sizes and paths checklist |

---

## 7. Verification of Deployment Constraints

1. **Model Weights Freeze**: All 16 personalized `.keras` files in `models/final_threshold_aware_fedper/` are verified intact with identical checksums.
2. **Feature Alignment**: The 472 model features strictly match `selected_features` (465 WAPs) followed by `signal_feature_names` (7 engineered statistics).
3. **RSSI Threshold Bounds**:
   - `Poor`: $\text{max RSSI} < -75\text{ dBm}$
   - `Fair`: $-75\text{ dBm} \le \text{max RSSI} < -65\text{ dBm}$
   - `Good`: $-65\text{ dBm} \le \text{max RSSI} < -55\text{ dBm}$
   - `Excellent`: $\text{max RSSI} \ge -55\text{ dBm}$
4. **Hybrid Decision Rule**: Margin $= 0.25\text{ dBm}$. Neural model invoked if and only if $\min_{t \in \{-75, -65, -55\}} |\text{max RSSI} - t| \le 0.25\text{ dBm}$ for a recognized client PHONEID; otherwise RSSI-rule fallback is executed.
