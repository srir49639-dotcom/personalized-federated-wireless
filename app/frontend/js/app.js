/**
 * Main Application Controller (Navigation, Theme, API fetching, Skeletons, Toasts)
 */

const App = {
  activeTab: "tab-overview",
  clientDataCache: [],

  init() {
    this.initTheme();
    this.initNavigation();
    PredictorUI.init();
    this.initClientSearch();
    this.loadAllData();
  },

  // --------------------------------------------------------------------------
  // Toast Notification System
  // --------------------------------------------------------------------------
  showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "✓" : type === "error" ? "✕" : "ℹ";
    toast.innerHTML = `<span style="font-weight: 800; font-size: 1rem;">${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(12px)";
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  },

  // --------------------------------------------------------------------------
  // Theme Management
  // --------------------------------------------------------------------------
  initTheme() {
    const savedTheme = localStorage.getItem("pfw_theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);

    const toggleBtn = document.getElementById("theme-toggle-btn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        const next = current === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("pfw_theme", next);
        this.showToast(`Switched to ${next === "dark" ? "Dark Space" : "Light Studio"} theme`, "info");
        // Trigger chart theme refresh
        this.refreshChartsTheme();
      });
    }
  },

  refreshChartsTheme() {
    this.loadAnalyticsData();
  },

  // --------------------------------------------------------------------------
  // Navigation & Interactive Tabs
  // --------------------------------------------------------------------------
  initNavigation() {
    const navItems = document.querySelectorAll(".nav-item[data-tab]");
    navItems.forEach(item => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        const targetTab = item.getAttribute("data-tab");
        this.switchTab(targetTab);

        // Close mobile drawer if open
        const sidebar = document.querySelector(".sidebar");
        if (sidebar) sidebar.classList.remove("open");
      });
    });

    // KPI Card Clickable Jump-Tabs
    document.querySelectorAll(".kpi-card[data-jump-tab]").forEach(card => {
      card.addEventListener("click", () => {
        const target = card.getAttribute("data-jump-tab");
        if (target) {
          this.switchTab(target);
          this.showToast(`Navigated to ${card.querySelector(".kpi-title")?.textContent.trim() || 'Tab'}`, "info");
        }
      });
    });

    const menuToggle = document.getElementById("menu-toggle-btn");
    if (menuToggle) {
      menuToggle.addEventListener("click", () => {
        const sidebar = document.querySelector(".sidebar");
        if (sidebar) sidebar.classList.toggle("open");
      });
    }
  },

  switchTab(tabId) {
    this.activeTab = tabId;

    // Update Nav item classes
    document.querySelectorAll(".nav-item[data-tab]").forEach(el => {
      el.classList.toggle("active", el.getAttribute("data-tab") === tabId);
    });

    // Show selected section
    document.querySelectorAll(".tab-content").forEach(el => {
      el.classList.toggle("active", el.id === tabId);
    });

    // Update title badge
    const activeNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (activeNav) {
      const titleSpan = activeNav.querySelector(".nav-label");
      const badge = document.getElementById("header-title-badge");
      if (badge && titleSpan) badge.textContent = titleSpan.textContent;
    }

    // Lazy load specific tab data if needed
    if (tabId === "tab-dataset") {
      this.loadDatasetExplorer(1);
    }
  },

  // --------------------------------------------------------------------------
  // Data Fetching & Rendering
  // --------------------------------------------------------------------------
  async loadAllData() {
    await Promise.all([
      this.loadKPIs(),
      this.loadAnalyticsData(),
      this.loadClientGrid()
    ]);
  },

  async loadKPIs() {
    try {
      const res = await fetch("/api/metrics");
      const kpis = await res.json();

      document.getElementById("kpi-acc").textContent = `${kpis.hybrid_test_accuracy}%`;
      document.getElementById("kpi-balanced").textContent = `${kpis.hybrid_balanced_accuracy}%`;
      document.getElementById("kpi-f1").textContent = `${kpis.hybrid_weighted_f1}%`;
      document.getElementById("kpi-clients").textContent = `${kpis.federated_clients}`;
      document.getElementById("kpi-features").textContent = `${kpis.model_features}`;
      document.getElementById("kpi-samples").textContent = `${kpis.dataset_samples.toLocaleString()}`;
    } catch (e) {
      console.error("Error loading KPIs:", e);
    }
  },

  async loadAnalyticsData() {
    try {
      // 1. Model comparison
      const compRes = await fetch("/api/model-comparison");
      const comparisonData = await compRes.json();
      ChartManager.initModelComparisonChart("chart-model-comparison", comparisonData);
      this.renderComparisonTable(comparisonData);

      // 2. Hybrid strategy summary
      const stratRes = await fetch("/api/hybrid-strategy");
      const stratData = await stratRes.json();
      ChartManager.initDecisionUsageDonut("chart-decision-donut", stratData);
      document.getElementById("strat-neural-pct").textContent = `${stratData.neural_usage_percentage}%`;
      document.getElementById("strat-rule-pct").textContent = `${stratData.rule_usage_percentage}%`;

      // 3. Training convergence
      const histRes = await fetch("/api/training-history");
      const historyData = await histRes.json();
      ChartManager.initConvergenceChart("chart-convergence-history", historyData);

      // 4. Confusion matrix & per class
      const cmRes = await fetch("/api/confusion-matrix");
      const cmData = await cmRes.json();
      this.renderConfusionMatrix(cmData);
      ChartManager.initPerClassChart("chart-per-class", cmData.per_class);

      // 5. Non-IID Heatmap
      const nonIidRes = await fetch("/api/non-iid-analysis");
      const nonIidData = await nonIidRes.json();
      this.renderNonIIDHeatmap(nonIidData);

    } catch (e) {
      console.error("Error loading analytics charts:", e);
    }
  },

  async loadClientGrid() {
    try {
      const res = await fetch("/api/client-metrics");
      this.clientDataCache = await res.json();

      // Chart
      ChartManager.initClientAccuracyChart("chart-client-accuracies", this.clientDataCache);

      // Render cards
      this.renderFilteredClients(this.clientDataCache);
    } catch (e) {
      console.error("Error loading client metrics:", e);
    }
  },

  initClientSearch() {
    const searchInput = document.getElementById("client-search-input");
    if (!searchInput) return;

    searchInput.addEventListener("input", (e) => {
      const query = e.target.value.trim().toLowerCase();
      if (!query) {
        this.renderFilteredClients(this.clientDataCache);
        return;
      }
      const filtered = this.clientDataCache.filter(c => 
        String(c.phone_id).includes(query) || `phoneid ${c.phone_id}`.includes(query)
      );
      this.renderFilteredClients(filtered);
    });
  },

  renderFilteredClients(clients) {
    const container = document.getElementById("clients-grid-container");
    const countLabel = document.getElementById("clients-counter-label");
    if (!container) return;

    container.innerHTML = "";

    if (countLabel) {
      countLabel.textContent = `Showing ${clients.length} of ${this.clientDataCache.length} Personalized Devices`;
    }

    if (clients.length === 0) {
      container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 30px; color: var(--text-dim);">No personalized devices matching search query.</div>`;
      return;
    }

    clients.forEach(c => {
      const card = document.createElement("div");
      card.className = "client-card";
      card.innerHTML = `
        <div>
          <div class="client-header">
            <span class="client-id-badge">📱 PHONEID ${c.phone_id}</span>
            <span class="client-acc">${c.fedper_test_accuracy}% Acc</span>
          </div>
          <div class="client-details">
            Samples: <strong>${c.total_samples}</strong> (Train: ${c.train_samples} | Val: ${c.val_samples} | Test: ${c.test_samples})
          </div>
          <div style="background: rgba(255, 255, 255, 0.08); height: 7px; border-radius: 4px; overflow: hidden; margin-bottom: 12px;">
            <div style="width: ${c.fedper_test_accuracy}%; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald)); height: 100%; border-radius: 4px;"></div>
          </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
          <span style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Personalized Model</span>
          <button type="button" class="btn-outline-xs btn-test-device" data-phone="${c.phone_id}" title="Select this PHONEID in Live Predictor">
            ⚡ Test Device
          </button>
        </div>
      `;

      card.querySelector(".btn-test-device")?.addEventListener("click", (e) => {
        e.stopPropagation();
        this.selectClientInPredictor(c.phone_id);
      });

      card.addEventListener("click", () => {
        this.selectClientInPredictor(c.phone_id);
      });

      container.appendChild(card);
    });
  },

  selectClientInPredictor(phoneId) {
    this.switchTab("tab-live-prediction");
    const phoneSelect = document.getElementById("input-phone-id");
    if (phoneSelect) {
      phoneSelect.value = String(phoneId);
      this.showToast(`Selected PHONEID ${phoneId} for Live Prediction`, "info");
    }
  },

  renderComparisonTable(data) {
    const tbody = document.getElementById("model-comparison-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600;">
          ${row.model} ${row.is_hybrid ? '<span class="status-pill" style="padding: 2px 8px; font-size: 0.68rem; margin-left: 6px;">Locked</span>' : ""}
        </td>
        <td><strong style="color: var(--accent-emerald); font-family: var(--font-mono);">${row.accuracy}%</strong></td>
        <td style="font-family: var(--font-mono);">${row.balanced_accuracy}%</td>
        <td style="font-family: var(--font-mono);">${row.weighted_f1}%</td>
      `;
      tbody.appendChild(tr);
    });
  },

  renderConfusionMatrix(data) {
    const matrix = data.matrix_counts;
    const classes = data.class_names;
    const tbody = document.getElementById("confusion-matrix-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    matrix.forEach((row, rIdx) => {
      const tr = document.createElement("tr");
      let cells = `<td style="font-weight: 700; color: var(--text-muted);">${classes[rIdx]} (True)</td>`;
      row.forEach((count, cIdx) => {
        const isDiag = rIdx === cIdx;
        const bg = isDiag ? "rgba(16, 185, 129, 0.22)" : count > 0 ? "rgba(239, 68, 68, 0.16)" : "transparent";
        cells += `<td style="background: ${bg}; font-weight: ${isDiag ? "800" : "normal"}; font-family: var(--font-mono);">${count}</td>`;
      });
      tr.innerHTML = cells;
      tbody.appendChild(tr);
    });

    // Per class cards
    const classContainer = document.getElementById("per-class-cards");
    if (classContainer) {
      classContainer.innerHTML = "";
      data.per_class.forEach(pc => {
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="card-body">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
              <span class="quality-badge badge-${pc.class_name}" style="font-size: 0.85rem; padding: 4px 12px;">${pc.class_name}</span>
              <span style="font-size: 0.8rem; color: var(--text-dim);">${pc.support} samples</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-top: 8px; font-family: var(--font-mono);">
              <span>Precision: <strong>${pc.precision}%</strong></span>
              <span>Recall: <strong>${pc.recall}%</strong></span>
              <span>F1: <strong>${pc.f1_score}%</strong></span>
            </div>
          </div>
        `;
        classContainer.appendChild(div);
      });
    }
  },

  renderNonIIDHeatmap(data) {
    const tbody = document.getElementById("non-iid-heatmap-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.proportions_matrix.forEach(row => {
      const tr = document.createElement("tr");
      let cells = `<td style="font-weight: 700;">📱 PHONEID ${row.phone_id}</td>`;
      row.percentages.forEach(pct => {
        const alpha = Math.min(1.0, Math.max(0.08, pct / 100));
        const bg = `rgba(6, 182, 212, ${alpha})`;
        const textColor = alpha > 0.45 ? "#ffffff" : "var(--text-main)";
        cells += `<td style="background: ${bg}; color: ${textColor}; font-weight: 700; font-family: var(--font-mono);">${pct}%</td>`;
      });
      tr.innerHTML = cells;
      tbody.appendChild(tr);
    });
  },

  // --------------------------------------------------------------------------
  // Dataset Explorer (Paginated)
  // --------------------------------------------------------------------------
  async loadDatasetExplorer(page = 1) {
    const tbody = document.getElementById("dataset-explorer-tbody");
    const pageIndicator = document.getElementById("dataset-page-indicator");
    if (!tbody) return;

    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 20px;">Loading dataset records...</td></tr>`;

    try {
      const res = await fetch(`/api/dataset-samples?page=${page}&page_size=10`);
      const data = await res.json();

      tbody.innerHTML = "";
      data.rows.forEach(r => {
        const wapsText = Object.entries(r.top_detected_waps).map(([w, val]) => `${w}: ${val}dBm`).join(", ");
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>PHONEID ${r.phone_id}</strong></td>
          <td>Building ${r.building_id}</td>
          <td>Floor ${r.floor}</td>
          <td><span class="status-pill" style="padding: 2px 8px; font-size: 0.75rem;">${r.detected_waps_count} WAPs</span></td>
          <td><strong style="font-family: var(--font-mono);">${r.max_rssi_dbm} dBm</strong></td>
          <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted);">${wapsText}</td>
        `;
        tbody.appendChild(tr);
      });

      if (pageIndicator) {
        pageIndicator.textContent = `Page ${data.page} of ${data.total_pages} (${data.total_rows.toLocaleString()} total rows)`;
      }

      // Hook pagination buttons
      const prevBtn = document.getElementById("btn-dataset-prev");
      const nextBtn = document.getElementById("btn-dataset-next");
      if (prevBtn) {
        prevBtn.disabled = data.page <= 1;
        prevBtn.onclick = () => this.loadDatasetExplorer(data.page - 1);
      }
      if (nextBtn) {
        nextBtn.disabled = data.page >= data.total_pages;
        nextBtn.onclick = () => this.loadDatasetExplorer(data.page + 1);
      }
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="6" style="color: var(--accent-rose);">Failed to load records.</td></tr>`;
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  App.init();
});
