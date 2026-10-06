// State & Globals
const API_BASE = (window.location.protocol === 'file:' || (window.location.port && window.location.port !== '5000'))
  ? 'http://127.0.0.1:5000'
  : '';

let sampleData = { brain: [], oral: [] };

document.addEventListener("DOMContentLoaded", () => {
  fetchSampleImages();
  fetchModelStats();
  setupDragDrop("brain-dropzone", "brain-file-input", handleBrainFileUpload);
  setupDragDrop("oral-dropzone", "oral-file-input", handleOralFileUpload);
});

function setupDragDrop(zoneId, inputId, handler) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handler({ target: { files: [file] } });
  });
  // Prevent double-trigger from button inside zone
  const btn = zone.querySelector(".btn-upload");
  if (btn) btn.addEventListener("click", (e) => {
    e.stopPropagation();
    document.getElementById(inputId).click();
  });
}

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  const targetBtn = document.querySelector(`[onclick="switchTab('${tabId}')"]`);
  if (targetBtn) targetBtn.classList.add("active");

  const targetContent = document.getElementById(`tab-${tabId}`);
  if (targetContent) targetContent.classList.add("active");
}

// Fetch Sample Images for Quick Testing
async function fetchSampleImages() {
  try {
    const res = await fetch(`${API_BASE}/api/sample-images`);
    sampleData = await res.json();

    // Brain Samples
    const brainContainer = document.getElementById("brain-sample-container");
    if (brainContainer && sampleData.brain) {
      brainContainer.innerHTML = sampleData.brain.map(s => `
        <button class="btn-sample" onclick="runBrainSample('${s.filename}')">
          🧪 ${s.name}
        </button>
      `).join("");
    }

    // Oral Samples
    const oralContainer = document.getElementById("oral-sample-container");
    if (oralContainer && sampleData.oral) {
      oralContainer.innerHTML = sampleData.oral.map(s => `
        <button class="btn-sample" onclick="runOralSample('${s.filename}')">
          🧪 ${s.name}
        </button>
      `).join("");
    }
  } catch (err) {
    console.error("Failed to load sample images:", err);
  }
}

// ==================== BRAIN TUMOR LOGIC ====================
function handleBrainFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("brain-results", "Please upload a valid image file (JPG, PNG, etc.)");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("brain-preview-img").src = e.target.result;
    submitBrainPrediction(file, null);
  };
  reader.readAsDataURL(file);
}

function runBrainSample(filename) {
  document.getElementById("brain-preview-img").src = `${API_BASE}/api/sample-image/brain/${filename}`;
  submitBrainPrediction(null, filename);
}

function showError(containerId, message) {
  const container = document.getElementById(containerId);
  const placeholder = container.previousElementSibling;
  if (placeholder && placeholder.id && placeholder.id.includes("placeholder")) {
    placeholder.style.display = "none";
  }
  document.getElementById("brain-placeholder") && (document.getElementById("brain-placeholder").style.display = "none");
  document.getElementById("oral-placeholder") && (document.getElementById("oral-placeholder").style.display = "none");
  container.style.display = "block";
  container.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink); font-size:0.9rem;">⚠️ ${message}</div>`;
}

async function submitBrainPrediction(file, sampleFilename) {
  const scanLine = document.getElementById("brain-scan-line");
  const placeholder = document.getElementById("brain-placeholder");
  const resultsBox = document.getElementById("brain-results");

  scanLine.style.display = "block";
  placeholder.style.display = "none";
  resultsBox.style.display = "block";
  resultsBox.innerHTML = `<div style="text-align:center; padding:1.5rem; color:var(--text-muted);">🔄 Running AI inference... please wait</div>`;

  try {
    let res;
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      res = await fetch(`${API_BASE}/api/predict/brain-tumor`, { method: "POST", body: formData });
    } else {
      res = await fetch(`${API_BASE}/api/predict/brain-tumor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: sampleFilename })
      });
    }

    const data = await res.json();
    scanLine.style.display = "none";

    if (data.error) {
      resultsBox.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink);">⚠️ ${data.error}</div>`;
      return;
    }

    const severityClass = data.is_cancerous ? (data.severity === "High" ? "badge-danger" : "badge-warning") : "badge-normal";
    const probBars = Object.keys(data.probabilities).map(key => {
      const item = data.probabilities[key];
      return `
        <div class="prob-item">
          <div class="prob-meta">
            <span>${item.display}</span>
            <span><strong>${item.percentage}%</strong></span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: 0%" data-target="${item.percentage}"></div>
          </div>
        </div>
      `;
    }).join("");

    resultsBox.innerHTML = `
      <div class="result-header">
        <div id="brain-badge" class="diagnosis-badge ${severityClass}">${data.display_name}</div>
        <div>
          <span class="confidence-val" id="brain-confidence">${data.confidence}%</span>
          <span style="font-size:0.8rem; color:var(--text-muted);"> Confidence</span>
        </div>
      </div>
      <div class="prob-list" id="brain-prob-list">${probBars}</div>
      <div class="rec-box" id="brain-recommendation">${data.recommendation}</div>
    `;

    // Animate probability bars
    requestAnimationFrame(() => {
      resultsBox.querySelectorAll(".prob-fill[data-target]").forEach(bar => {
        bar.style.width = bar.getAttribute("data-target") + "%";
      });
    });

  } catch (err) {
    scanLine.style.display = "none";
    resultsBox.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink);">⚠️ Prediction request failed. Is the Flask server running?</div>`;
  }
}

// ==================== ORAL CANCER LOGIC ====================
function handleOralFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    const resultsBox = document.getElementById("oral-results");
    document.getElementById("oral-placeholder").style.display = "none";
    resultsBox.style.display = "block";
    resultsBox.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink);">⚠️ Please upload a valid image file.</div>`;
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("oral-preview-img").src = e.target.result;
    submitOralPrediction(file, null);
  };
  reader.readAsDataURL(file);
}

function runOralSample(filename) {
  document.getElementById("oral-preview-img").src = `${API_BASE}/api/sample-image/oral/${filename}`;
  submitOralPrediction(null, filename);
}

async function submitOralPrediction(file, sampleFilename) {
  const scanLine = document.getElementById("oral-scan-line");
  const placeholder = document.getElementById("oral-placeholder");
  const resultsBox = document.getElementById("oral-results");

  scanLine.style.display = "block";
  placeholder.style.display = "none";
  resultsBox.style.display = "block";
  resultsBox.innerHTML = `<div style="text-align:center; padding:1.5rem; color:var(--text-muted);">🔄 Running AI inference... please wait</div>`;

  try {
    let res;
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      res = await fetch(`${API_BASE}/api/predict/oral-cancer`, { method: "POST", body: formData });
    } else {
      res = await fetch(`${API_BASE}/api/predict/oral-cancer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: sampleFilename })
      });
    }

    const data = await res.json();
    scanLine.style.display = "none";

    if (data.error) {
      resultsBox.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink);">⚠️ ${data.error}</div>`;
      return;
    }

    const badgeClass = data.is_cancerous ? "badge-danger" : "badge-normal";
    resultsBox.innerHTML = `
      <div class="result-header">
        <div class="diagnosis-badge ${badgeClass}">${data.result_str}</div>
        <div>
          <span class="confidence-val">${data.confidence}%</span>
          <span style="font-size:0.8rem; color:var(--text-muted);"> Probability</span>
        </div>
      </div>
      <div class="rec-box" style="margin-top:1rem;">${data.recommendation}</div>
    `;

  } catch (err) {
    scanLine.style.display = "none";
    resultsBox.innerHTML = `<div style="background:rgba(255,75,92,0.1); border:1px solid rgba(255,75,92,0.4); border-radius:10px; padding:1rem; color:var(--danger-pink);">⚠️ Oral cancer prediction failed. Is the Flask server running?</div>`;
  }
}

// ==================== CERVICAL CANCER LOGIC ====================
function toggleSmokingFields() {
  const isSmoker = document.getElementById("inp-smokes").checked;
  document.getElementById("smoking-fields").style.display = isSmoker ? "block" : "none";
}

function toggleHcFields() {
  const hasHc = document.getElementById("inp-hc").checked;
  document.getElementById("hc-fields").style.display = hasHc ? "block" : "none";
}

function loadCervicalPreset(type) {
  if (type === 'low') {
    document.getElementById('inp-age').value = 24;
    document.getElementById('inp-partners').value = 1;
    document.getElementById('inp-first-sex').value = 19;
    document.getElementById('inp-pregnancies').value = 1;
    document.getElementById('inp-smokes').checked = false;
    document.getElementById('inp-hc').checked = false;
    document.getElementById('inp-stds').checked = false;
    document.getElementById('inp-dx-cancer').checked = false;
  } else if (type === 'moderate') {
    document.getElementById('inp-age').value = 35;
    document.getElementById('inp-partners').value = 3;
    document.getElementById('inp-first-sex').value = 17;
    document.getElementById('inp-pregnancies').value = 2;
    document.getElementById('inp-smokes').checked = true;
    document.getElementById('inp-smoke-years').value = 5;
    document.getElementById('inp-hc').checked = true;
    document.getElementById('inp-hc-years').value = 3;
    document.getElementById('inp-stds').checked = false;
    document.getElementById('inp-dx-cancer').checked = false;
  } else if (type === 'high') {
    document.getElementById('inp-age').value = 45;
    document.getElementById('inp-partners').value = 6;
    document.getElementById('inp-first-sex').value = 14;
    document.getElementById('inp-pregnancies').value = 4;
    document.getElementById('inp-smokes').checked = true;
    document.getElementById('inp-smoke-years').value = 18;
    document.getElementById('inp-hc').checked = true;
    document.getElementById('inp-hc-years').value = 10;
    document.getElementById('inp-stds').checked = true;
    document.getElementById('inp-dx-cancer').checked = true;
  }

  // Update slider labels
  document.getElementById('lbl-age').innerText = document.getElementById('inp-age').value;
  document.getElementById('lbl-partners').innerText = document.getElementById('inp-partners').value;
  document.getElementById('lbl-first-sex').innerText = document.getElementById('inp-first-sex').value;
  document.getElementById('lbl-pregnancies').innerText = document.getElementById('inp-pregnancies').value;
  document.getElementById('lbl-smoke-years').innerText = document.getElementById('inp-smoke-years').value || 0;
  document.getElementById('lbl-hc-years').innerText = document.getElementById('inp-hc-years').value || 0;

  toggleSmokingFields();
  toggleHcFields();
  runCervicalPrediction();
}

async function runCervicalPrediction() {
  const payload = {
    "Age": parseInt(document.getElementById("inp-age").value),
    "Number of sexual partners": parseInt(document.getElementById("inp-partners").value),
    "First sexual intercourse": parseInt(document.getElementById("inp-first-sex").value),
    "Num of pregnancies": parseInt(document.getElementById("inp-pregnancies").value),
    "Smokes": document.getElementById("inp-smokes").checked ? 1 : 0,
    "Smokes (years)": document.getElementById("inp-smokes").checked ? parseInt(document.getElementById("inp-smoke-years").value) : 0,
    "Hormonal Contraceptives": document.getElementById("inp-hc").checked ? 1 : 0,
    "Hormonal Contraceptives (years)": document.getElementById("inp-hc").checked ? parseInt(document.getElementById("inp-hc-years").value) : 0,
    "STDs": document.getElementById("inp-stds").checked ? 1 : 0,
    "STDs:HPV": document.getElementById("inp-stds").checked ? 1 : 0,
    "Dx:Cancer": document.getElementById("inp-dx-cancer").checked ? 1 : 0,
    "Dx:HPV": document.getElementById("inp-dx-cancer").checked ? 1 : 0
  };

  const placeholder = document.getElementById("cervical-placeholder");
  const resultsBox = document.getElementById("cervical-results");

  placeholder.style.display = "none";
  resultsBox.style.display = "block";

  try {
    const res = await fetch(`${API_BASE}/api/predict/cervical-cancer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    const badge = document.getElementById("cervical-badge");
    badge.innerText = data.risk_status;
    badge.className = "diagnosis-badge " + (data.is_high_risk ? "badge-danger" : "badge-normal");

    document.getElementById("cervical-risk-val").innerText = `${data.risk_percentage}%`;
    document.getElementById("cervical-risk-fill").style.width = `${data.risk_percentage}%`;

    // Sub-models breakdown
    const mBreakdown = document.getElementById("cervical-model-breakdown");
    mBreakdown.innerHTML = Object.keys(data.individual_models).map(m => `
      <div style="background:rgba(30,41,59,0.5); padding:0.4rem 0.6rem; border-radius:6px; display:flex; justify-space-between;">
        <span><strong>${m}:</strong></span>
        <span style="color:${data.individual_models[m].includes('Positive') ? 'var(--danger-pink)' : 'var(--success-green)'}">
          ${data.individual_models[m]}
        </span>
      </div>
    `).join("");

    // Risk factors
    const factorsList = document.getElementById("cervical-factors-list");
    factorsList.innerHTML = data.key_risk_factors.map(f => `<li>${f}</li>`).join("");

    // Recommendation
    document.getElementById("cervical-recommendation").innerText = data.recommendation;

  } catch (err) {
    alert("Cervical cancer evaluation failed.");
  }
}

// ==================== MODEL METRICS ====================
async function fetchModelStats() {
  const container = document.getElementById("stats-container");
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/model-stats`);
    const stats = await res.json();

    const cardMap = [
      { key: "brain_tumor", title: "🧠 Brain Tumor Classifier", icon: "🧠" },
      { key: "oral_cancer", title: "👄 Oral Cancer Detector", icon: "👄" },
      { key: "cervical_cancer", title: "🩸 Cervical Risk Ensemble", icon: "🩸" }
    ];

    container.innerHTML = cardMap.map(item => {
      const s = stats[item.key];
      return `
        <div class="stat-card">
          <div class="stat-header">${item.icon} ${item.title}</div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Algorithm:</span>
            <span><strong>${s.algorithm}</strong></span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Test Accuracy:</span>
            <span style="color:var(--primary-cyan); font-weight:700;">${s.accuracy}</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">F1-Score:</span>
            <span>${s.f1_score}</span>
          </div>
          <div class="metric-row">
            <span style="color:var(--text-muted);">Precision:</span>
            <span>${s.precision}</span>
          </div>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load model stats:", err);
  }
}
