// Point this at your backend. If you're opening this via the FastAPI
// static mount (http://localhost:8000/), this can stay as-is.
const API_BASE = window.location.origin.includes("8000")
  ? ""
  : "http://localhost:8000";

// Your second website — food-scanning app — runs on port 8001.
const FOODSCAN_BASE_URL = "http://localhost:8001";

const inputs = {
  eyeWhole: document.getElementById("eye-whole-input"),
  eyeLid: document.getElementById("eye-lid-input"),
  nail: document.getElementById("nail-input"),
  tongue: document.getElementById("tongue-input"),
};

const previews = {
  eyeWhole: document.getElementById("eye-whole-preview"),
  eyeLid: document.getElementById("eye-lid-preview"),
  nail: document.getElementById("nail-preview"),
  tongue: document.getElementById("tongue-preview"),
};

const files = { eyeWhole: null, eyeLid: null, nail: null, tongue: null };

Object.entries(inputs).forEach(([key, input]) => {
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    files[key] = file;
    const url = URL.createObjectURL(file);
    previews[key].src = url;
    previews[key].style.display = "block";
    previews[key].nextElementSibling.style.display = "none";
  });
});

const analyzeBtn = document.getElementById("analyze-btn");
const statusMsg = document.getElementById("status-msg");
const resultsSection = document.getElementById("results");
const summaryGrid = document.getElementById("summary-cards");
const detailGrid = document.getElementById("detail-cards");
const recommendationsSection = document.getElementById("food-recommendations-section");
const recommendationsGrid = document.getElementById("food-recommendations");
const downloadReportBtn = document.getElementById("download-report-btn");

// Holds everything needed to (re)generate the PDF report, filled in by
// renderResults() after each successful scan.
let lastReport = null;

const NUTRIENT_LABELS = {
  vitamin_a: "Vitamin A",
  vitamin_b: "Vitamin B",
  iron: "Iron",
  zinc: "Zinc",
};

const NUTRIENT_THEME = {
  vitamin_a: { icon: "👁️", cardClass: "food-card-a" },
  vitamin_b: { icon: "⚡", cardClass: "food-card-b" },
  iron: { icon: "🩸", cardClass: "food-card-iron" },
  zinc: { icon: "🛡️", cardClass: "food-card-zinc" },
};

// ------------------------------------------------------------------
// Food recommendations per nutrient.
// ------------------------------------------------------------------
const FOOD_DB = {
  iron: {
    veg: ["dal", "chickpeas", "rajma", "tofu", "spinach", "methi", "pumpkin seeds", "sesame seeds"],
    nonveg: ["chicken liver", "mutton", "eggs", "chicken", "fish like sardines and mackerel"],
    tips: [
      "Pair iron-rich plant foods with vitamin-C-rich foods such as lemon, amla, guava, oranges, or tomatoes to improve iron absorption.",
      "Keep tea and coffee away from iron-rich meals because they can reduce iron absorption.",
    ],
  },
  vitamin_a: {
    veg: ["carrots", "sweet potatoes", "spinach and other leafy greens", "mango", "papaya", "milk and dairy"],
    nonveg: ["eggs", "chicken liver", "fish like salmon and mackerel"],
    tips: [
      "Vitamin A is fat-soluble, so having these foods with a little ghee, oil, or nuts helps your body absorb it better.",
    ],
  },
  vitamin_b: {
    veg: ["whole grains", "lentils", "beans", "bananas", "leafy greens", "nuts", "seeds"],
    nonveg: ["eggs", "chicken", "fish like salmon and tuna", "mutton", "liver"],
    tips: [
      "Choose a variety of these foods regularly to support adequate vitamin B intake.",
    ],
  },
  zinc: {
    veg: ["pumpkin seeds", "chickpeas", "lentils", "nuts like cashews and almonds", "whole grains", "dairy and yogurt"],
    nonveg: ["chicken", "mutton", "eggs", "fish", "prawns and crab"],
    tips: [
      "Soaking or sprouting grains and legumes before cooking reduces phytates, which can otherwise block zinc absorption.",
    ],
  },
};

// Currently selected food preference: "veg" or "nonveg".
let selectedDiet = "veg";

function getFoodsForDiet(db, diet) {
  return diet === "nonveg" ? [...db.nonveg, ...db.veg] : db.veg;
}

function renderFoodRecommendations(flaggedNutrients) {
  recommendationsGrid.innerHTML = "";

  if (flaggedNutrients.length === 0) {
    recommendationsSection.style.display = "none";
    return;
  }

  flaggedNutrients.forEach((nutrient) => {
    const db = FOOD_DB[nutrient];
    if (!db) return;

    const theme = NUTRIENT_THEME[nutrient] || { icon: "🍽️", cardClass: "" };

    const chips = getFoodsForDiet(db, selectedDiet)
      .map((f) => `<span class="food-chip">${f}</span>`)
      .join("");

    const tips = db.tips
      .map((t) => `<li><span class="tip-icon">💡</span><span>${t}</span></li>`)
      .join("");

    const card = document.createElement("div");
    card.className = `food-card ${theme.cardClass}`;
    card.innerHTML = `
      <div class="food-card-header">
        <span class="food-card-icon">${theme.icon}</span>
        <h4>${NUTRIENT_LABELS[nutrient] || nutrient}</h4>
      </div>
      <div class="food-chip-row">${chips}</div>
      <ul class="food-tips">${tips}</ul>
    `;
    recommendationsGrid.appendChild(card);
  });

  recommendationsSection.style.display = "block";
}

// Any class at or above this probability is treated as a "significant"
// sign for explanation purposes (nail / tongue).
const SIGNIFICANT_THRESHOLD = 0.20;

// ------------------------------------------------------------------
// Plain-language meaning of each class, per model.
// ------------------------------------------------------------------
const SIGN_INFO = {
  eye: {
    vitamin_a_deficient: { label: "signs linked to vitamin A deficiency", nutrient: "Vitamin A" },
    iron_deficient: { label: "signs linked to iron deficiency", nutrient: "Iron" },
    normal: { label: "no concerning signs", nutrient: null },
  },
  nail: {
    Healthy: { label: "healthy-looking nails", nutrient: null },
    Leukonychia: { label: "white spots or patches (leukonychia)", nutrient: "Zinc" },
    Pale: { label: "paleness", nutrient: "Iron" },
    Spoon: { label: "spoon-shaped nails", nutrient: "Iron" },
  },
  tongue: {
    red: { label: "redness", nutrient: "Vitamin B" },
    cracked: { label: "cracks or fissures", nutrient: "Vitamin B" },
    pale: { label: "paleness", nutrient: "Iron" },
    healthy: { label: "no concerning signs", nutrient: null },
  },
};

// ------------------------------------------------------------------
// Special combination rules, checked BEFORE the generic explanation.
// ------------------------------------------------------------------
const COMBO_RULES = {
  tongue: [
    {
      match: (sigClasses) => sigClasses.includes("pale") && sigClasses.includes("healthy"),
      text: () =>
        "Your tongue shows a mix of healthy and pale areas — this can be an early sign of iron deficiency that hasn't fully developed yet. Worth keeping an eye on.",
    },
  ],
  nail: [],
  eye: [],
};

function joinWithAnd(items) {
  if (items.length === 1) return items[0];
  return items.slice(0, -1).join(", ") + " and " + items.slice(-1);
}

function getSignificantClasses(probabilities) {
  return Object.entries(probabilities)
    .filter(([, p]) => p >= SIGNIFICANT_THRESHOLD)
    .sort((a, b) => b[1] - a[1])
    .map(([c]) => c);
}

// For the eye, the backend already applied the model's own thresholds
// (detected_conditions), so use those instead of the generic 20% cutoff.
function getSignificantClassesFor(modelKey, result) {
  if (modelKey === "eye" && Array.isArray(result.detected_conditions)) {
    return result.detected_conditions;
  }
  return getSignificantClasses(result.probabilities);
}

function buildExplanation(modelKey, result) {
  const signInfo = SIGN_INFO[modelKey];
  if (!signInfo) return null;

  const sigClasses = getSignificantClassesFor(modelKey, result);

  // Check combo rules first
  for (const rule of COMBO_RULES[modelKey] || []) {
    if (rule.match(sigClasses)) {
      return rule.text();
    }
  }

  // Generic explanation from significant, meaningful (non-"healthy") signs
  const meaningful = sigClasses
    .map((c) => ({ cls: c, info: signInfo[c] }))
    .filter((s) => s.info && s.info.nutrient);

  if (meaningful.length === 0) {
    return "No concerning signs detected — this looks healthy.";
  }

  const labels = joinWithAnd(meaningful.map((s) => s.info.label));
  const nutrients = joinWithAnd([...new Set(meaningful.map((s) => s.info.nutrient))]);

  return `This shows ${labels}, which can be associated with ${nutrients} deficiency.`;
}

// ------------------------------------------------------------------
// Decides what counts as "the prediction(s)" to show in Model Details.
// - tongue and eye: every detected condition with its own confidence;
//   if none, show "healthy" using healthy_probability.
// - nail: just the one winning class.
// ------------------------------------------------------------------
function getTopPredictions(modelKey, result) {
  if ((modelKey === "tongue" || modelKey === "eye") && Array.isArray(result.detected_conditions)) {
    if (result.detected_conditions.length > 0) {
      return result.detected_conditions.map((c) => ({
        label: c,
        prob: result.probabilities[c],
      }));
    }
    return [{ label: "healthy", prob: result.healthy_probability }];
  }
  return [{ label: result.prediction, prob: result.confidence }];
}

analyzeBtn.addEventListener("click", async () => {
  if (!files.eyeWhole && !files.eyeLid && !files.nail && !files.tongue) {
    statusMsg.textContent = "Upload at least one image first.";
    return;
  }

  // Eye scan needs both photos (whole eye + pulled-down lower eyelid)
  if ((files.eyeWhole && !files.eyeLid) || (!files.eyeWhole && files.eyeLid)) {
    statusMsg.textContent =
      "For the eye scan, please upload both photos (whole eye + pulled-down lower eyelid).";
    return;
  }

  analyzeBtn.disabled = true;
  statusMsg.textContent = "Analyzing images…";
  resultsSection.style.display = "none";

  const formData = new FormData();
  if (files.eyeWhole) formData.append("eye_whole_file", files.eyeWhole);
  if (files.eyeLid) formData.append("eye_lid_file", files.eyeLid);
  if (files.nail) formData.append("nail_file", files.nail);
  if (files.tongue) formData.append("tongue_file", files.tongue);

  try {
    const res = await fetch(`${API_BASE}/predict/all`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResults(data);
    statusMsg.textContent = "";
  } catch (err) {
    statusMsg.textContent = `Error: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ------------------------------------------------------------------
// Works out which scans ACTUALLY show a significant sign for a nutrient,
// so the summary and the details never disagree.
// ------------------------------------------------------------------
function getContributingModels(nutrient, individualPredictions) {
  const label = NUTRIENT_LABELS[nutrient];
  const contributors = [];

  Object.entries(individualPredictions || {}).forEach(([modelKey, result]) => {
    if (!result || !result.probabilities) return;
    const signInfo = SIGN_INFO[modelKey] || {};
    const hasSign = getSignificantClassesFor(modelKey, result).some(
      (c) => signInfo[c] && signInfo[c].nutrient === label
    );
    if (hasSign) contributors.push(modelKey);
  });

  return contributors;
}

function renderResults(data) {
  summaryGrid.innerHTML = "";
  detailGrid.innerHTML = "";

  const flaggedNutrients = [];

  // Summary cards - ONLY nutrients that came back flagged as deficient.
  Object.entries(data.deficiency_summary).forEach(([nutrient, info]) => {
    if (!info.likely_deficient) return;

    flaggedNutrients.push(nutrient);
    const pct = Math.round(info.score * 100);
    const contributors = getContributingModels(nutrient, data.individual_predictions);
    const basedOn = (contributors.length ? contributors : [...new Set(info.based_on)]).join(", ") || "no data";

    const card = document.createElement("div");
    card.className = "nutrient-card";
    card.innerHTML = `
      <h4>
        ${NUTRIENT_LABELS[nutrient] || nutrient}
        <span class="badge warn">Possible deficiency</span>
      </h4>
      <div class="bar-track">
        <div class="bar-fill warn" style="width:${pct}%"></div>
      </div>
      <small>${pct}% confidence · based on ${basedOn}</small>
    `;
    summaryGrid.appendChild(card);
  });

  if (flaggedNutrients.length === 0) {
    summaryGrid.innerHTML = `
      <div class="no-deficiency">
        ✅ No deficiencies detected from the uploaded images.
      </div>
    `;
  }

  // Detail cards - one per model that ran
  const preds = data.individual_predictions;
  Object.entries(preds).forEach(([modelKey, result]) => {
    if (!result) return;
    const card = document.createElement("div");
    card.className = "detail-card";

    const explanation = buildExplanation(modelKey, result);
    const signsNote = explanation ? `<p class="signs-note">${explanation}</p>` : "";

    const topPreds = getTopPredictions(modelKey, result);
    const predLabel = topPreds.length > 1 ? "Top predictions" : "Top prediction";
    const predText = topPreds
      .map((p) => `<strong>${p.label}</strong> (${(p.prob * 100).toFixed(1)}%)`)
      .join(", ");

    card.innerHTML = `
      <h4>${modelKey.charAt(0).toUpperCase() + modelKey.slice(1)} model</h4>
      ${signsNote}
      <p>${predLabel}: ${predText}</p>
    `;
    detailGrid.appendChild(card);
  });

  renderFoodRecommendations(flaggedNutrients);

  lastReport = {
    deficiencySummary: data.deficiency_summary,
    individualPredictions: data.individual_predictions,
    flaggedNutrients,
  };

  resultsSection.style.display = "block";
}

// ------------------------------------------------------------------
// PDF report generation (jsPDF + autoTable, loaded via CDN in scan.html)
// ------------------------------------------------------------------
function generateReportPDF() {
  if (!lastReport) return;

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 40;
  let y = 50;

  const PRIMARY = [108, 79, 214];
  const TEXT_DIM = [107, 100, 128];
  const WARN = [232, 89, 12];
  const OK = [47, 158, 68];

  // ---- Header ----
  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.setTextColor(...PRIMARY);
  doc.text("NutriScan AI — Deficiency Report", margin, y);
  y += 20;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(...TEXT_DIM);
  const generatedOn = new Date().toLocaleString();
  doc.text(`Generated on ${generatedOn}`, margin, y);
  y += 25;

  // ---- Deficiency Summary table ----
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(0, 0, 0);
  doc.text("Deficiency Summary", margin, y);
  y += 10;

  const flaggedEntries = Object.entries(lastReport.deficiencySummary)
    .filter(([, info]) => info.likely_deficient);

  const summaryRows = flaggedEntries.length
    ? flaggedEntries.map(([nutrient, info]) => {
        const contributors = getContributingModels(nutrient, lastReport.individualPredictions);
        const basedOn = (contributors.length ? contributors : [...new Set(info.based_on)]).join(", ") || "no data";
        return [
          NUTRIENT_LABELS[nutrient] || nutrient,
          "Possible deficiency",
          `${Math.round(info.score * 100)}%`,
          basedOn,
        ];
      })
    : [["—", "No deficiencies detected", "—", "—"]];

  doc.autoTable({
    startY: y,
    margin: { left: margin, right: margin },
    head: [["Nutrient", "Status", "Confidence", "Based on"]],
    body: summaryRows,
    theme: "grid",
    headStyles: { fillColor: PRIMARY, textColor: 255, fontStyle: "bold" },
    styles: { fontSize: 10, cellPadding: 6 },
    didParseCell: (data) => {
      if (data.section === "body" && data.column.index === 1) {
        data.cell.styles.textColor = data.cell.raw === "Possible deficiency" ? WARN : OK;
        data.cell.styles.fontStyle = "bold";
      }
    },
  });
  y = doc.lastAutoTable.finalY + 25;

  // ---- Model Details ----
  const ensureSpace = (needed) => {
    if (y + needed > doc.internal.pageSize.getHeight() - margin) {
      doc.addPage();
      y = 50;
    }
  };

  ensureSpace(30);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(0, 0, 0);
  doc.text("Model Details", margin, y);
  y += 18;

  Object.entries(lastReport.individualPredictions).forEach(([modelKey, result]) => {
    if (!result) return;
    ensureSpace(60);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...PRIMARY);
    doc.text(`${modelKey.charAt(0).toUpperCase() + modelKey.slice(1)} model`, margin, y);
    y += 15;

    const explanation = buildExplanation(modelKey, result);
    if (explanation) {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.setTextColor(...TEXT_DIM);
      const lines = doc.splitTextToSize(explanation, pageWidth - margin * 2);
      doc.text(lines, margin, y);
      y += lines.length * 13 + 4;
    }

    const topPreds = getTopPredictions(modelKey, result);
    const predLabel = topPreds.length > 1 ? "Top predictions" : "Top prediction";
    const predText = topPreds.map((p) => `${p.label} (${(p.prob * 100).toFixed(1)}%)`).join(", ");

    ensureSpace(20);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    doc.text(`${predLabel}: ${predText}`, margin, y);
    y += 22;
  });

  // ---- Food Recommendations ----
  if (lastReport.flaggedNutrients.length > 0) {
    ensureSpace(30);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(0, 0, 0);
    doc.text(
      `Food Recommendations (${selectedDiet === "nonveg" ? "Non-Vegetarian" : "Vegetarian"})`,
      margin,
      y
    );
    y += 18;

    lastReport.flaggedNutrients.forEach((nutrient) => {
      const db = FOOD_DB[nutrient];
      if (!db) return;

      ensureSpace(40);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.setTextColor(...PRIMARY);
      doc.text(NUTRIENT_LABELS[nutrient] || nutrient, margin, y);
      y += 15;

      const foodSentence = `Include ${joinWithAnd(getFoodsForDiet(db, selectedDiet))} in your diet.`;
      [foodSentence, ...db.tips].forEach((line) => {
        ensureSpace(20);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.setTextColor(...TEXT_DIM);
        const wrapped = doc.splitTextToSize(`•  ${line}`, pageWidth - margin * 2);
        doc.text(wrapped, margin, y);
        y += wrapped.length * 13 + 4;
      });
      y += 8;
    });
  }

  // ---- Disclaimer footer on every page ----
  const pageCount = doc.internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setDrawColor(228, 223, 242);
    doc.line(margin, pageHeight - 50, pageWidth - margin, pageHeight - 50);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(8.5);
    doc.setTextColor(...TEXT_DIM);
    doc.text(
      "This report is generated by an AI screening tool and is not a medical diagnosis. Please share it with a doctor before making any changes to your diet, supplements, or routine.",
      margin,
      pageHeight - 35,
      { maxWidth: pageWidth - margin * 2 }
    );
    doc.text(`Page ${i} of ${pageCount}`, pageWidth - margin - 60, pageHeight - 20);
  }

  const filename = `NutriScan-Report-${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}

downloadReportBtn.addEventListener("click", generateReportPDF);

// ------------------------------------------------------------------
// Veg / Non-veg toggle
// ------------------------------------------------------------------
document.querySelectorAll(".diet-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    selectedDiet = btn.dataset.diet;

    document.querySelectorAll(".diet-btn").forEach((b) => {
      b.classList.toggle("active", b === btn);
    });

    if (lastReport) {
      renderFoodRecommendations(lastReport.flaggedNutrients);
    }
  });
});