/**
 * Charts and Data Visualization Module (Chart.js Integration)
 */

const ChartManager = {
  charts: {},

  // Get theme-aware colors
  getColors() {
    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    return {
      textColor: isDark ? "#94a3b8" : "#475569",
      gridColor: isDark ? "rgba(255, 255, 255, 0.07)" : "rgba(0, 0, 0, 0.06)",
      tooltipBg: isDark ? "#1e293b" : "#ffffff",
      tooltipText: isDark ? "#f8fafc" : "#0f172a",
      primary: "#10b981",
      secondary: "#06b6d4",
      purple: "#8b5cf6",
      indigo: "#6366f1",
      amber: "#f59e0b",
      rose: "#ef4444",
    };
  },

  destroyChart(id) {
    if (this.charts[id]) {
      this.charts[id].destroy();
      delete this.charts[id];
    }
  },

  // 1. Model Comparison Chart
  initModelComparisonChart(canvasId, comparisonData) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = this.getColors();
    const labels = comparisonData.map(d => d.model);
    const accuracies = comparisonData.map(d => d.accuracy);
    const balanced = comparisonData.map(d => d.balanced_accuracy);
    const f1Scores = comparisonData.map(d => d.weighted_f1);

    this.charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Test Accuracy (%)",
            data: accuracies,
            backgroundColor: "rgba(16, 185, 129, 0.85)",
            borderRadius: 6,
          },
          {
            label: "Balanced Accuracy (%)",
            data: balanced,
            backgroundColor: "rgba(6, 182, 212, 0.85)",
            borderRadius: 6,
          },
          {
            label: "Weighted F1 (%)",
            data: f1Scores,
            backgroundColor: "rgba(139, 92, 246, 0.85)",
            borderRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: colors.textColor } },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipText,
            bodyColor: colors.tooltipText,
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1
          }
        },
        scales: {
          x: {
            ticks: { color: colors.textColor },
            grid: { color: colors.gridColor }
          },
          y: {
            min: 75,
            max: 100,
            ticks: { color: colors.textColor },
            grid: { color: colors.gridColor }
          }
        }
      }
    });
  },

  // 2. Decision Routing Donut
  initDecisionUsageDonut(canvasId, strategySummary) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = this.getColors();
    this.charts[canvasId] = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["RSSI Rule Decision", "Personalized Neural Model"],
        datasets: [{
          data: [
            strategySummary.rule_usage_percentage,
            strategySummary.neural_usage_percentage
          ],
          backgroundColor: ["#0ea5e9", "#f59e0b"],
          borderWidth: 2,
          borderColor: colors.tooltipBg
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { color: colors.textColor } }
        },
        cutout: "68%"
      }
    });
  },

  // 3. Convergence History Chart
  initConvergenceChart(canvasId, historyData) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = this.getColors();
    const fedper = historyData.fedper_final || [];
    const rounds = fedper.map(d => `R${d.round}`);
    const trainAcc = fedper.map(d => d.train_acc);
    const valAcc = fedper.map(d => d.mean_client_val);
    const pooledVal = fedper.map(d => d.pooled_val);

    this.charts[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: rounds,
        datasets: [
          {
            label: "Train Accuracy (%)",
            data: trainAcc,
            borderColor: colors.indigo,
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            tension: 0.3,
            fill: false,
            borderWidth: 2.2
          },
          {
            label: "Mean Client Val Acc (%)",
            data: valAcc,
            borderColor: colors.primary,
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            tension: 0.3,
            fill: false,
            borderWidth: 2.2
          },
          {
            label: "Pooled Val Acc (%)",
            data: pooledVal,
            borderColor: colors.amber,
            borderDash: [5, 5],
            tension: 0.3,
            fill: false,
            borderWidth: 1.8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: colors.textColor } }
        },
        scales: {
          x: { ticks: { color: colors.textColor }, grid: { color: colors.gridColor } },
          y: { ticks: { color: colors.textColor }, grid: { color: colors.gridColor } }
        }
      }
    });
  },

  // 4. Per-Client Accuracy Chart
  initClientAccuracyChart(canvasId, clientsData) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = this.getColors();
    const labels = clientsData.map(c => `ID ${c.phone_id}`);
    const accs = clientsData.map(c => c.fedper_test_accuracy);

    this.charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "FedPer Test Accuracy (%)",
          data: accs,
          backgroundColor: "rgba(16, 185, 129, 0.8)",
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: colors.textColor } }
        },
        scales: {
          x: { ticks: { color: colors.textColor, maxRotation: 45 }, grid: { color: colors.gridColor } },
          y: { min: 60, max: 100, ticks: { color: colors.textColor }, grid: { color: colors.gridColor } }
        }
      }
    });
  },

  // 5. Per-Class Radar / Bar Chart
  initPerClassChart(canvasId, perClassData) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const colors = this.getColors();
    const labels = perClassData.map(d => d.class_name);
    const prec = perClassData.map(d => d.precision);
    const rec = perClassData.map(d => d.recall);
    const f1 = perClassData.map(d => d.f1_score);

    this.charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          { label: "Precision (%)", data: prec, backgroundColor: "#3b82f6", borderRadius: 4 },
          { label: "Recall (%)", data: rec, backgroundColor: "#10b981", borderRadius: 4 },
          { label: "F1 Score (%)", data: f1, backgroundColor: "#8b5cf6", borderRadius: 4 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: colors.textColor } }
        },
        scales: {
          x: { ticks: { color: colors.textColor }, grid: { color: colors.gridColor } },
          y: { min: 90, max: 102, ticks: { color: colors.textColor }, grid: { color: colors.gridColor } }
        }
      }
    });
  }
};
