/**
 * Live Prediction Controller & Interactive RSSI Meter (V2.5 Ultra)
 */

const DEMO_SAMPLES = {
  boundary_sample: {
    phone_id: "1",
    label: "Boundary Margin Sample",
    desc: "Boundary Sample (max_rssi ~ -64.8 dBm within 0.25 dBm margin of -65 dBm boundary)",
    primary_rssi: -64.8,
    rssi: `WAP001=-64.8
WAP002=-70.0
WAP005=-75.0
WAP010=-82.0
WAP020=-85.0`
  },
  excellent_sample: {
    phone_id: "14",
    label: "Strong / Excellent Sample",
    desc: "Strong Signal (max_rssi ~ -48.0 dBm in Excellent region)",
    primary_rssi: -48.0,
    rssi: `WAP001=-48.0
WAP012=-54.0
WAP015=-61.0
WAP019=-72.0`
  },
  fair_sample: {
    phone_id: "13",
    label: "Moderate / Fair Sample",
    desc: "Moderate Signal (max_rssi ~ -71.5 dBm in Fair region)",
    primary_rssi: -71.5,
    rssi: `WAP001=-71.5
WAP003=-76.0
WAP008=-80.0
WAP014=-84.0`
  },
  poor_sample: {
    phone_id: "7",
    label: "Weak / Poor Sample",
    desc: "Weak Signal (max_rssi ~ -82.0 dBm in Poor region)",
    primary_rssi: -82.0,
    rssi: `WAP001=-82.0
WAP046=-86.0
WAP050=-89.0
WAP055=-92.0`
  }
};

const PredictorUI = {
  init() {
    const btnPredict = document.getElementById("btn-predict");
    const btnSample = document.getElementById("btn-load-sample");
    const btnClear = document.getElementById("btn-clear-input");
    const btnToggleJson = document.getElementById("btn-toggle-json");
    const btnCopyJson = document.getElementById("btn-copy-json");
    const slider = document.getElementById("interactive-rssi-slider");

    if (btnPredict) btnPredict.addEventListener("click", () => this.runPrediction());
    if (btnSample) btnSample.addEventListener("click", () => this.loadDemoSample("boundary_sample"));
    if (btnClear) btnClear.addEventListener("click", () => this.clearInput());

    // Preset chips
    document.querySelectorAll(".preset-chip[data-preset]").forEach(chip => {
      chip.addEventListener("click", () => {
        const presetKey = chip.getAttribute("data-preset");
        this.loadDemoSample(presetKey);
      });
    });

    // Real-time Slider Listener
    if (slider) {
      slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        this.onSliderChange(val);
      });
    }

    // Toggle JSON & Copy JSON
    if (btnToggleJson) {
      btnToggleJson.addEventListener("click", () => {
        const wrapper = document.getElementById("json-code-wrapper");
        if (wrapper) {
          const isHidden = wrapper.style.display === "none";
          wrapper.style.display = isHidden ? "block" : "none";
          btnToggleJson.innerHTML = isHidden ? "<span>{ }</span> Hide Raw JSON" : "<span>{ }</span> Toggle Raw JSON";
        }
      });
    }

    if (btnCopyJson) {
      btnCopyJson.addEventListener("click", () => {
        const codeElem = document.getElementById("res-json-code");
        if (codeElem && codeElem.textContent) {
          navigator.clipboard.writeText(codeElem.textContent).then(() => {
            if (window.App && App.showToast) {
              App.showToast("Output JSON copied to clipboard!", "success");
            } else {
              alert("Copied to clipboard!");
            }
          });
        }
      });
    }

    // Initialize needle & slider to default
    this.updateMeter(-64.8);
  },

  onSliderChange(rssiVal) {
    const display = document.getElementById("slider-rssi-val");
    if (display) display.textContent = `${rssiVal.toFixed(1)} dBm`;

    this.updateMeter(rssiVal);

    // Update or insert WAP001 in textarea
    const textarea = document.getElementById("input-rssi-measurements");
    if (textarea) {
      const lines = textarea.value.split("\n");
      let found = false;
      const updatedLines = lines.map(line => {
        if (line.trim().toUpperCase().startsWith("WAP001=")) {
          found = true;
          return `WAP001=${rssiVal.toFixed(1)}`;
        }
        return line;
      });
      if (!found) {
        updatedLines.unshift(`WAP001=${rssiVal.toFixed(1)}`);
      }
      textarea.value = updatedLines.join("\n");
    }
  },

  loadDemoSample(type = "boundary_sample") {
    const sample = DEMO_SAMPLES[type] || DEMO_SAMPLES.boundary_sample;
    const phoneSelect = document.getElementById("input-phone-id");
    const rssiTextarea = document.getElementById("input-rssi-measurements");
    const slider = document.getElementById("interactive-rssi-slider");
    const sliderDisplay = document.getElementById("slider-rssi-val");

    if (phoneSelect) phoneSelect.value = sample.phone_id;
    if (rssiTextarea) rssiTextarea.value = sample.rssi;
    if (slider) slider.value = sample.primary_rssi;
    if (sliderDisplay) sliderDisplay.textContent = `${sample.primary_rssi.toFixed(1)} dBm`;

    this.updateMeter(sample.primary_rssi);

    if (window.App && App.showToast) {
      App.showToast(`Loaded ${sample.label} (${sample.primary_rssi} dBm)`, "info");
    }
  },

  clearInput() {
    const rssiTextarea = document.getElementById("input-rssi-measurements");
    if (rssiTextarea) rssiTextarea.value = "";
    document.getElementById("prediction-result-panel").style.display = "none";
    document.getElementById("prediction-empty-state").style.display = "block";
    this.updateMeter(-85);

    const slider = document.getElementById("interactive-rssi-slider");
    const sliderDisplay = document.getElementById("slider-rssi-val");
    if (slider) slider.value = -85;
    if (sliderDisplay) sliderDisplay.textContent = "-85.0 dBm";

    if (window.App && App.showToast) {
      App.showToast("Inputs cleared.", "info");
    }
  },

  updateMeter(maxRssi) {
    const needle = document.getElementById("meter-needle");
    const needleLabel = document.getElementById("meter-needle-label");
    if (!needle) return;

    // Range: -100 dBm (0%) to -30 dBm (100%)
    const clamped = Math.max(-100, Math.min(-30, maxRssi));
    const pct = ((clamped - (-100)) / 70) * 100;

    needle.style.left = `${pct}%`;
    if (needleLabel) {
      needleLabel.textContent = `${maxRssi.toFixed(1)} dBm`;
    }
  },

  async runPrediction() {
    const phoneSelect = document.getElementById("input-phone-id");
    const rssiTextarea = document.getElementById("input-rssi-measurements");
    const btnPredict = document.getElementById("btn-predict");

    const phoneId = phoneSelect.value === "unknown" ? null : parseInt(phoneSelect.value);
    const rawRssi = rssiTextarea.value.trim();

    if (!rawRssi) {
      if (window.App && App.showToast) {
        App.showToast("Please enter at least one WAP RSSI measurement.", "error");
      } else {
        alert("Please enter at least one WAP RSSI measurement (e.g. WAP001=-55).");
      }
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = `<span class="pulse-dot"></span> Predicting...`;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phone_id: phoneId,
          rssi: rawRssi
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction request failed.");
      }

      const res = await response.json();
      this.displayResult(res);

      if (window.App && App.showToast) {
        App.showToast(`Predicted: ${res.class_name} (${(res.confidence * 100).toFixed(1)}%)`, "success");
      }
    } catch (e) {
      if (window.App && App.showToast) {
        App.showToast("Error: " + e.message, "error");
      } else {
        alert("Error: " + e.message);
      }
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<span>⚡</span> Predict Link Quality`;
    }
  },

  displayResult(data) {
    document.getElementById("prediction-empty-state").style.display = "none";
    const panel = document.getElementById("prediction-result-panel");
    panel.style.display = "block";

    // Update Meter
    const maxRssi = data.signal_stats.max_rssi;
    this.updateMeter(maxRssi);

    // Class badge
    const badge = document.getElementById("res-class-badge");
    badge.className = `quality-badge badge-${data.class_name}`;
    badge.textContent = data.class_name;

    // Confidence
    const confPct = (data.confidence * 100).toFixed(1);
    document.getElementById("res-confidence").textContent = `${confPct}%`;
    const confBar = document.getElementById("res-confidence-bar");
    if (confBar) {
      confBar.style.width = `${confPct}%`;
    }

    document.getElementById("res-description").textContent = data.class_description;

    // Decision Flowchart Step Highlights
    const flowRssi = document.getElementById("flow-val-rssi");
    const flowMargin = document.getElementById("flow-val-margin");
    const flowRoute = document.getElementById("flow-val-route");
    const card2 = document.getElementById("flow-card-2");
    const card3 = document.getElementById("flow-card-3");

    if (flowRssi) flowRssi.textContent = `${maxRssi.toFixed(1)} dBm`;

    const inMargin = data.distance_to_threshold_dbm <= 0.25;
    if (flowMargin) {
      flowMargin.textContent = inMargin ? `In Margin (${data.distance_to_threshold_dbm.toFixed(2)} dBm)` : `Clear (${data.distance_to_threshold_dbm.toFixed(2)} dBm)`;
    }

    if (card2) {
      card2.className = inMargin ? "flow-step-card active" : "flow-step-card";
    }

    if (flowRoute && card3) {
      if (data.used_neural) {
        flowRoute.textContent = `Personalized DL Head (Client ${data.phone_id || 'Fallback'})`;
        card3.className = "flow-step-card active-neural";
      } else {
        flowRoute.textContent = "Deterministic Physical Rule";
        card3.className = "flow-step-card active-rule";
      }
    }

    // Decision reason
    document.getElementById("res-decision-reason").textContent = data.decision_reason;

    // Model & Neural usage pill
    document.getElementById("res-model-used").textContent = data.model_used;
    const neuralPill = document.getElementById("res-neural-pill");
    if (data.used_neural) {
      neuralPill.textContent = "Yes (Personalized Head)";
      neuralPill.className = "status-pill";
      neuralPill.style.color = "#10b981";
      neuralPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
    } else {
      neuralPill.textContent = "No (Deterministic Rule)";
      neuralPill.className = "status-pill";
      neuralPill.style.color = "#06b6d4";
      neuralPill.style.borderColor = "rgba(6, 182, 212, 0.4)";
    }

    // Signal stats
    document.getElementById("res-max-rssi").textContent = `${maxRssi.toFixed(1)} dBm`;
    document.getElementById("res-mean-rssi").textContent = `${data.signal_stats.mean_rssi.toFixed(1)} dBm`;
    document.getElementById("res-detected-aps").textContent = `${data.signal_stats.detected_aps}`;
    document.getElementById("res-threshold-dist").textContent = `${data.distance_to_threshold_dbm.toFixed(2)} dBm`;

    // Raw JSON modal/code
    const codeElem = document.getElementById("res-json-code");
    if (codeElem) {
      codeElem.textContent = JSON.stringify(data, null, 2);
    }
  }
};
