"""Generate All Research Analytics Figures.

Generates 18 publication-quality research figures directly from actual project
datasets, round histories, model comparison tables, and test predictions.
All figures are saved to graphs/generated/.
"""

import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.autolayout"] = True

# Resolve directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_DIR = PROJECT_ROOT / "dataset" / "UJIndoorLoc"
OUTPUT_DIR = PROJECT_ROOT / "graphs" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Poor", "Fair", "Good", "Excellent"]
CLASS_COLORS = ["#ef4444", "#f59e0b", "#06b6d4", "#10b981"]

print(f"Generating analytics figures in: {OUTPUT_DIR}")

# ----------------------------------------------------------------------------
# Load Core Data
# ----------------------------------------------------------------------------
print("Loading datasets and test predictions...")
with open(PREPROCESSING_DIR / "federated_data_3way_final.pkl", "rb") as f:
    fed_data = pickle.load(f)

pred_df = pd.read_csv(RESULTS_DIR / "FINAL_HYBRID_TEST_PREDICTIONS.csv")
client_res_df = pd.read_csv(RESULTS_DIR / "final_fedper_client_test_results.csv")
comp_df = pd.read_csv(RESULTS_DIR / "FINAL_RESULTS_COMPARISON.csv")

# Extract full labels and features from federated splits
all_labels = np.concatenate([fed_data["y_train"], fed_data["y_val"], fed_data["y_test"]])
unique_clients = fed_data["unique_clients"]
all_clients = np.concatenate([fed_data["clients_train"], fed_data["clients_val"], fed_data["clients_test"]])

# Raw training data for RSSI distributions
train_csv_path = DATASET_DIR / "trainingData.csv"
train_df = pd.read_csv(train_csv_path) if train_csv_path.exists() else None

# ----------------------------------------------------------------------------
# 1. 01_dataset_class_distribution.png
# ----------------------------------------------------------------------------
print("Generating 01_dataset_class_distribution.png...")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
counts = [np.sum(all_labels == i) for i in range(4)]
total = len(all_labels)
bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="black", linewidth=1.2, alpha=0.9)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = (count / total) * 100
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 100, f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("UJIIndoorLoc Dataset Class Distribution (19,937 Federated Samples)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_ylim(0, max(counts) * 1.18)
fig.savefig(OUTPUT_DIR / "01_dataset_class_distribution.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. 02_rssi_distribution.png
# ----------------------------------------------------------------------------
print("Generating 02_rssi_distribution.png...")
if train_df is not None:
    wap_cols = [c for c in train_df.columns if c.startswith("WAP")]
    all_rssi = train_df[wap_cols].values.flatten()
    detected_rssi = all_rssi[(all_rssi > -100) & (all_rssi < 0)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(detected_rssi, bins=60, kde=True, color="#3b82f6", edgecolor="none", alpha=0.7, ax=ax)
    ax.axvline(-75, color="#ef4444", linestyle="--", linewidth=1.5, label="Poor / Fair Boundary (-75 dBm)")
    ax.axvline(-65, color="#f59e0b", linestyle="--", linewidth=1.5, label="Fair / Good Boundary (-65 dBm)")
    ax.axvline(-55, color="#10b981", linestyle="--", linewidth=1.5, label="Good / Excellent Boundary (-55 dBm)")
    ax.set_title("Empirical Distribution of Detected Wi-Fi RSSI Values (dBm)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Signal Strength (dBm)", fontsize=11)
    ax.set_ylabel("Detection Frequency", fontsize=11)
    ax.legend(frameon=True)
    fig.savefig(OUTPUT_DIR / "02_rssi_distribution.png")
    plt.close(fig)

# ----------------------------------------------------------------------------
# 3. 03_max_rssi_distribution.png
# ----------------------------------------------------------------------------
print("Generating 03_max_rssi_distribution.png...")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sns.histplot(pred_df["max_rssi_dbm"], bins=50, kde=True, color="#6366f1", edgecolor="white", alpha=0.8, ax=ax)
ax.axvline(-75, color="#ef4444", linestyle="--", linewidth=1.8, label="Poor / Fair (-75 dBm)")
ax.axvline(-65, color="#f59e0b", linestyle="--", linewidth=1.8, label="Fair / Good (-65 dBm)")
ax.axvline(-55, color="#10b981", linestyle="--", linewidth=1.8, label="Good / Excellent (-55 dBm)")
ax.set_title("Distribution of Maximum RSSI on Held-Out Test Set (N = 2,998)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Maximum Observed RSSI (dBm)", fontsize=11)
ax.set_ylabel("Sample Count", fontsize=11)
ax.legend(frameon=True)
fig.savefig(OUTPUT_DIR / "03_max_rssi_distribution.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. 04_detected_aps_distribution.png
# ----------------------------------------------------------------------------
print("Generating 04_detected_aps_distribution.png...")
if train_df is not None:
    wap_cols = [c for c in train_df.columns if c.startswith("WAP")]
    detected_counts = (train_df[wap_cols] != 100).sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(detected_counts, bins=40, kde=True, color="#0ea5e9", edgecolor="white", alpha=0.85, ax=ax)
    ax.set_title("Distribution of Detected Access Points (APs) per Fingerprint Sample", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Concurrently Detected WAPs", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.axvline(detected_counts.median(), color="#dc2626", linestyle=":", linewidth=2, label=f"Median APs: {detected_counts.median():.0f}")
    ax.legend(frameon=True)
    fig.savefig(OUTPUT_DIR / "04_detected_aps_distribution.png")
    plt.close(fig)

# ----------------------------------------------------------------------------
# 5. 05_client_sample_distribution.png
# ----------------------------------------------------------------------------
print("Generating 05_client_sample_distribution.png...")
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
train_counts = pd.Series(fed_data["clients_train"]).value_counts().sort_index()
val_counts = pd.Series(fed_data["clients_val"]).value_counts().sort_index()
test_counts = pd.Series(fed_data["clients_test"]).value_counts().sort_index()
x_labels = [f"ID {pid}" for pid in unique_clients]
x_indices = np.arange(len(unique_clients))
width = 0.65

p1 = ax.bar(x_indices, train_counts, width, label="Train (70%)", color="#2563eb", edgecolor="black", linewidth=0.8)
p2 = ax.bar(x_indices, val_counts, width, bottom=train_counts, label="Validation (15%)", color="#06b6d4", edgecolor="black", linewidth=0.8)
p3 = ax.bar(x_indices, test_counts, width, bottom=train_counts + val_counts, label="Test (15%)", color="#f59e0b", edgecolor="black", linewidth=0.8)

ax.set_title("Sample Allocation per Federated Client (Non-IID Partition)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Total Number of Fingerprint Samples", fontsize=11)
ax.legend(frameon=True)
fig.savefig(OUTPUT_DIR / "05_client_sample_distribution.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. 06_client_class_distribution.png
# ----------------------------------------------------------------------------
print("Generating 06_client_class_distribution.png...")
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
client_class_counts = []
for c_idx in range(len(unique_clients)):
    mask = np.array(all_clients) == c_idx
    lbls = all_labels[mask]
    counts = [np.sum(lbls == k) for k in range(4)]
    client_class_counts.append(counts)

class_arr = np.array(client_class_counts)
bottoms = np.zeros(len(unique_clients))
for k in range(4):
    ax.bar(x_indices, class_arr[:, k], bottom=bottoms, label=CLASS_NAMES[k], color=CLASS_COLORS[k], edgecolor="black", linewidth=0.6)
    bottoms += class_arr[:, k]

ax.set_title("Non-IID Class Composition Across 16 Client Devices", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Sample Count", fontsize=11)
ax.legend(title="Link Quality Class", frameon=True)
fig.savefig(OUTPUT_DIR / "06_client_class_distribution.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. 07_client_class_heatmap.png
# ----------------------------------------------------------------------------
print("Generating 07_client_class_heatmap.png...")
proportions = class_arr / class_arr.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
sns.heatmap(proportions, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=CLASS_NAMES,
            yticklabels=[f"PHONEID {p}" for p in unique_clients], cbar_kws={"label": "Proportion Within Client (%)"}, ax=ax)
ax.set_title("Non-IID Class Heterogeneity Matrix (Client × Class %)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Wireless Link Quality Class", fontsize=11)
ax.set_ylabel("Federated Client Device", fontsize=11)
fig.savefig(OUTPUT_DIR / "07_client_class_heatmap.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 8. 08_client_test_accuracy.png
# ----------------------------------------------------------------------------
print("Generating 08_client_test_accuracy.png...")
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
accs = client_res_df["Accuracy"] * 100
bars = ax.bar(x_indices, accs, width=0.65, color="#10b981", edgecolor="black", linewidth=0.8)
mean_acc = accs.mean()
ax.axhline(mean_acc, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean FedPer Accuracy: {mean_acc:.2f}%")
for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_title("Personalized FedPer Test Accuracy Across All 16 Clients", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(frameon=True)
fig.savefig(OUTPUT_DIR / "08_client_test_accuracy.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 9. 09_model_accuracy_comparison.png
# ----------------------------------------------------------------------------
print("Generating 09_model_accuracy_comparison.png...")
models = ["Centralized MLP", "FedAvg", "Personalized FedPer", "Locked Hybrid"]
accuracies = [93.13, 82.89, 89.06, 97.93]
colors = ["#3b82f6", "#ef4444", "#8b5cf6", "#10b981"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Comparative Test Accuracy Across Architectures", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Held-Out Test Accuracy (%)", fontsize=11)
ax.set_ylim(70, 105)
fig.savefig(OUTPUT_DIR / "09_model_accuracy_comparison.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 10. 10_model_f1_comparison.png
# ----------------------------------------------------------------------------
print("Generating 10_model_f1_comparison.png...")
f1_scores = [93.13, 82.65, 89.02, 97.94]
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bars = ax.bar(models, f1_scores, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{f1:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Comparative Weighted F1-Score Across Architectures", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Weighted F1 Score (%)", fontsize=11)
ax.set_ylim(70, 105)
fig.savefig(OUTPUT_DIR / "10_model_f1_comparison.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 11. 11_fedper_convergence.png
# ----------------------------------------------------------------------------
print("Generating 11_fedper_convergence.png...")
fedper_hist_path = RESULTS_DIR / "final_fedper_round_history.csv"
if fedper_hist_path.exists():
    df_fph = pd.read_csv(fedper_hist_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_fph["round"], df_fph["train_accuracy"] * 100, label="Training Accuracy", color="#2563eb", linewidth=2.2)
    ax.plot(df_fph["round"], df_fph["mean_client_validation"] * 100, label="Mean Client Val Accuracy", color="#10b981", linewidth=2.2)
    ax.plot(df_fph["round"], df_fph["pooled_validation"] * 100, label="Pooled Val Accuracy", color="#f59e0b", linestyle="--", linewidth=2)
    ax.set_title("Federated Personalization (FedPer) Convergence History", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Federated Communication Rounds", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.legend(frameon=True)
    fig.savefig(OUTPUT_DIR / "11_fedper_convergence.png")
    plt.close(fig)

# ----------------------------------------------------------------------------
# 12. 12_validation_accuracy_history.png
# ----------------------------------------------------------------------------
print("Generating 12_validation_accuracy_history.png...")
fedavg_hist_path = RESULTS_DIR / "fedavg_round_history.csv"
if fedavg_hist_path.exists() and fedper_hist_path.exists():
    df_fah = pd.read_csv(fedavg_hist_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_fah["round"], df_fah["validation_accuracy"] * 100, label="FedAvg Global Model", color="#ef4444", linewidth=2.2, linestyle=":")
    ax.plot(df_fph["round"], df_fph["mean_client_validation"] * 100, label="FedPer Personalized Heads", color="#8b5cf6", linewidth=2.2)
    ax.set_title("Validation Accuracy Progression: FedAvg vs FedPer", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Communication Rounds", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.legend(frameon=True)
    fig.savefig(OUTPUT_DIR / "12_validation_accuracy_history.png")
    plt.close(fig)

# ----------------------------------------------------------------------------
# 13. 13_final_confusion_matrix.png
# ----------------------------------------------------------------------------
print("Generating 13_final_confusion_matrix.png...")
cm = confusion_matrix(pred_df["actual_class"], pred_df["hybrid_prediction"], labels=[0, 1, 2, 3])
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=ax)
ax.set_title("Final Hybrid Evaluation Confusion Matrix (N = 2,998)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Link Quality", fontsize=11, fontweight="bold")
ax.set_ylabel("True Link Quality", fontsize=11, fontweight="bold")
fig.savefig(OUTPUT_DIR / "13_final_confusion_matrix.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 14. 14_final_per_class_metrics.png
# ----------------------------------------------------------------------------
print("Generating 14_final_per_class_metrics.png...")
cr = classification_report(pred_df["actual_class"], pred_df["hybrid_prediction"], target_names=CLASS_NAMES, output_dict=True)
prec = [cr[c]["precision"] * 100 for c in CLASS_NAMES]
rec = [cr[c]["recall"] * 100 for c in CLASS_NAMES]
f1 = [cr[c]["f1-score"] * 100 for c in CLASS_NAMES]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x = np.arange(4)
width = 0.25
ax.bar(x - width, prec, width, label="Precision", color="#3b82f6", edgecolor="black", linewidth=0.8)
ax.bar(x, rec, width, label="Recall", color="#10b981", edgecolor="black", linewidth=0.8)
ax.bar(x + width, f1, width, label="F1-Score", color="#8b5cf6", edgecolor="black", linewidth=0.8)

ax.set_title("Per-Class Precision, Recall, and F1 Performance", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight="bold")
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(90, 103)
ax.legend(frameon=True)
fig.savefig(OUTPUT_DIR / "14_final_per_class_metrics.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 15. 15_hybrid_decision_usage.png
# ----------------------------------------------------------------------------
print("Generating 15_hybrid_decision_usage.png...")
used_neural = pred_df["used_neural"].sum()
used_rule = len(pred_df) - used_neural
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
wedges, texts, autotexts = ax.pie(
    [used_rule, used_neural],
    labels=["RSSI Threshold Rule", "Personalized Neural Model"],
    autopct="%1.1f%%",
    colors=["#0ea5e9", "#f59e0b"],
    startangle=140,
    explode=(0, 0.1),
    wedgeprops={"edgecolor": "black", "linewidth": 1.2},
)
for at in autotexts:
    at.set_color("black")
    at.set_fontweight("bold")
ax.set_title("Hybrid Inference Decision Routing (N = 2,998)", fontsize=13, fontweight="bold", pad=12)
fig.savefig(OUTPUT_DIR / "15_hybrid_decision_usage.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 16. 16_feature_pipeline.png
# ----------------------------------------------------------------------------
print("Generating 16_feature_pipeline.png...")
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.axis("off")
boxes = [
    ("Raw Wi-Fi Fingerprint\n(520 WAPs)", "#e2e8f0"),
    ("Feature Selection\n(Remove 55 constant)\n-> 465 Usable WAPs", "#bfdbfe"),
    ("Signal Feature Engineering\n(7 Stats: max, mean, APs...)", "#fef08a"),
    ("Concatenation\n(Exact 472 Features)", "#fed7aa"),
    ("MinMaxScaler\n([0, 1] Normalized)", "#bbf7d0"),
    ("Threshold-Aware FedPer\n(Hybrid Inference)", "#e9d5ff"),
]
from matplotlib.patches import FancyBboxPatch

for i, (text, color) in enumerate(boxes):
    rect = FancyBboxPatch((i * 1.6, 0.2), 1.35, 0.6, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(i * 1.6 + 0.675, 0.5, text, ha="center", va="center", fontsize=8.5, fontweight="bold")
    if i < len(boxes) - 1:
        ax.annotate("", xy=((i + 1) * 1.6, 0.5), xytext=(i * 1.6 + 1.35, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))
ax.set_xlim(-0.2, len(boxes) * 1.6)
ax.set_ylim(0, 1.0)
ax.set_title("472-Dimensional Feature Engineering & Preprocessing Flow", fontsize=13, fontweight="bold")
fig.savefig(OUTPUT_DIR / "16_feature_pipeline.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 17. 17_rssi_threshold_visualization.png
# ----------------------------------------------------------------------------
print("Generating 17_rssi_threshold_visualization.png...")
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
ax.set_xlim(-100, -30)
ax.set_ylim(0, 1)
ax.axvspan(-100, -75, color="#ef4444", alpha=0.3, label="Poor (< -75 dBm)")
ax.axvspan(-75, -65, color="#f59e0b", alpha=0.3, label="Fair (-75 to -65 dBm)")
ax.axvspan(-65, -55, color="#06b6d4", alpha=0.3, label="Good (-65 to -55 dBm)")
ax.axvspan(-55, -30, color="#10b981", alpha=0.3, label="Excellent (>= -55 dBm)")

# Annotate margin zones
for t in [-75, -65, -55]:
    ax.axvspan(t - 0.25, t + 0.25, color="#8b5cf6", alpha=0.7)
    ax.axvline(t, color="black", linestyle="--", linewidth=1.5)
    ax.text(t, 0.82, f"{t} dBm\n(+/-0.25 dBm Hybrid)", ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.9))

ax.text(-87.5, 0.4, "POOR", fontsize=12, fontweight="bold", ha="center", color="#991b1b")
ax.text(-70, 0.4, "FAIR", fontsize=12, fontweight="bold", ha="center", color="#92400e")
ax.text(-60, 0.4, "GOOD", fontsize=12, fontweight="bold", ha="center", color="#0e7490")
ax.text(-42.5, 0.4, "EXCELLENT", fontsize=12, fontweight="bold", ha="center", color="#065f46")

ax.set_yticks([])
ax.set_xlabel("Maximum Received Signal Strength Indication (dBm)", fontsize=11, fontweight="bold")
ax.set_title("RSSI Decision Boundaries & Locked Hybrid Margins (0.25 dBm)", fontsize=13, fontweight="bold", pad=12)
fig.savefig(OUTPUT_DIR / "17_rssi_threshold_visualization.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 18. 18_model_architecture.png
# ----------------------------------------------------------------------------
print("Generating 18_model_architecture.png...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis("off")

arch_elements = [
    ("Input Vector\n(472 Dimensions)", (0.1, 0.5), "#e2e8f0"),
    ("Shared Trunk\nDense(256)->LN->DO(0.08)\nDense(256)->LN->DO(0.08)\nDense(128)->LN->Dense(64)", (0.4, 0.7), "#bfdbfe"),
    ("Max RSSI Extract\nLambda Layer [pos 465]\nDense(16)->LN", (0.4, 0.45), "#fef08a"),
    ("Soft Thresholds\n3 Sigmoid Branches\n40*(t - thresh)", (0.4, 0.2), "#fed7aa"),
    ("Feature Fusion\nConcatenate Layer\n(64 + 16 + 3 = 83-dim)", (0.7, 0.5), "#bbf7d0"),
    ("Personalized Private Head\nDense(128)->DO(0.04)\nDense(64)", (0.9, 0.7), "#e9d5ff"),
    ("Softmax Classifier\n4 Quality Classes", (0.9, 0.3), "#fbcfe8"),
]

for name, (cx, cy), col in arch_elements:
    rect = FancyBboxPatch((cx - 0.12, cy - 0.1), 0.24, 0.2, boxstyle="round,pad=0.03", facecolor=col, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(cx, cy, name, ha="center", va="center", fontsize=8, fontweight="bold")

# Connect with arrows
ax.annotate("", xy=(0.28, 0.7), xytext=(0.22, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.28, 0.45), xytext=(0.22, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.28, 0.25), xytext=(0.22, 0.45), arrowprops=dict(arrowstyle="->", lw=1.5))

ax.annotate("", xy=(0.58, 0.52), xytext=(0.52, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.58, 0.5), xytext=(0.52, 0.45), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.58, 0.48), xytext=(0.52, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))

ax.annotate("", xy=(0.78, 0.7), xytext=(0.7, 0.6), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.9, 0.4), xytext=(0.9, 0.6), arrowprops=dict(arrowstyle="->", lw=1.5))

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.05, 0.95)
ax.set_title("Step 16 Threshold-Aware FedPer Neural Network Architecture", fontsize=14, fontweight="bold", pad=12)
fig.savefig(OUTPUT_DIR / "18_model_architecture.png")
plt.close(fig)

print("All 18 analytics figures generated successfully!")
