// State Management
let selectedFile = null;
let webcamStream = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  setupDragAndDrop();
  checkApiHealth();
  loadSampleImages();
});

// Theme Toggle (Bright Side Default <-> Dark Mode)
function initTheme() {
  const savedTheme = localStorage.getItem("app-theme") || "light";
  setTheme(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  setTheme(newTheme);
}

function setTheme(theme) {
  const btn = document.getElementById("theme-toggle-btn");
  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    if (btn) btn.innerHTML = `<i class="fa-solid fa-sun"></i> Light Theme`;
  } else {
    document.documentElement.removeAttribute("data-theme");
    if (btn) btn.innerHTML = `<i class="fa-solid fa-moon"></i> Dark Mode`;
  }
  localStorage.setItem("app-theme", theme);
}

// Check API Health
async function checkApiHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const badge = document.getElementById("status-badge");
    if (data.status === "online") {
      badge.innerHTML = `<span class="status-dot"></span> Model Ready (${data.device.toUpperCase()})`;
    }
  } catch (err) {
    console.warn("API health check warning:", err);
  }
}

// Switch Tabs
function switchTab(tabName) {
  const tabs = ["upload", "webcam", "samples"];
  tabs.forEach(t => {
    document.getElementById(`tab-${t}`).classList.remove("active");
    document.getElementById(`content-${t}`).classList.add("hidden");
  });
  document.getElementById(`tab-${tabName}`).classList.add("active");
  document.getElementById(`content-${tabName}`).classList.remove("hidden");

  if (tabName !== "webcam" && webcamStream) {
    stopWebcam();
  }
}

// Setup Drag & Drop
function setupDragAndDrop() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });
}

function handleSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    alert("Please select a valid image file.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("image-preview").src = e.target.result;
    document.querySelector(".drop-zone-prompt").classList.add("hidden");
    document.getElementById("preview-wrapper").classList.remove("hidden");
    document.getElementById("analyze-btn").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

function clearSelectedImage() {
  selectedFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("image-preview").src = "";
  document.querySelector(".drop-zone-prompt").classList.remove("hidden");
  document.getElementById("preview-wrapper").classList.add("hidden");
  document.getElementById("analyze-btn").classList.add("hidden");
}

// Predict Uploaded Image
async function predictUploadedImage() {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append("file", selectedFile);
  await sendPredictionRequest("/api/predict", {
    method: "POST",
    body: formData
  });
}

// Webcam Controls
async function startWebcam() {
  const video = document.getElementById("webcam-video");
  const overlay = document.getElementById("webcam-overlay");
  const startBtn = document.getElementById("start-webcam-btn");
  const captureBtn = document.getElementById("capture-btn");

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = webcamStream;
    overlay.classList.add("hidden");
    startBtn.classList.add("hidden");
    captureBtn.classList.remove("hidden");
  } catch (err) {
    alert("Could not access webcam: " + err.message);
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  const video = document.getElementById("webcam-video");
  if (video) video.srcObject = null;
  document.getElementById("webcam-overlay").classList.remove("hidden");
  document.getElementById("start-webcam-btn").classList.remove("hidden");
  document.getElementById("capture-btn").classList.add("hidden");
}

async function captureWebcam() {
  const video = document.getElementById("webcam-video");
  const canvas = document.getElementById("webcam-canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64Image = canvas.toDataURL("image/jpeg", 0.9);
  await sendPredictionRequest("/api/predict-base64", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: base64Image })
  });
}

// Load Sample Images Grid
async function loadSampleImages() {
  const grid = document.getElementById("samples-grid");
  try {
    const res = await fetch("/api/sample-images");
    const data = await res.json();
    
    if (data.samples && data.samples.length > 0) {
      grid.innerHTML = "";
      data.samples.forEach(sample => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.innerHTML = `
          <img src="${sample.image_b64}" alt="Sample ${sample.name}">
          <div class="sample-info">${sample.true_gender}, ${sample.true_age}y</div>
        `;
        card.onclick = () => predictSampleImage(sample.image_b64);
        grid.appendChild(card);
      });
    } else {
      grid.innerHTML = "<p class='section-desc'>No samples available.</p>";
    }
  } catch (err) {
    grid.innerHTML = "<p class='section-desc'>Failed to load sample images.</p>";
  }
}

async function predictSampleImage(base64Image) {
  await sendPredictionRequest("/api/predict-base64", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: base64Image })
  });
}

// Core Prediction Request Handler
async function sendPredictionRequest(url, fetchOptions) {
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsContent = document.getElementById("results-content");

  emptyState.classList.add("hidden");
  resultsContent.classList.add("hidden");
  loadingState.classList.remove("hidden");

  try {
    const res = await fetch(url, fetchOptions);
    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || "Prediction request failed.");
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Error: " + err.message);
    emptyState.classList.remove("hidden");
  } finally {
    loadingState.classList.add("hidden");
  }
}

// Render Results Dashboard
function renderResults(data) {
  const emptyState = document.getElementById("empty-state");
  const resultsContent = document.getElementById("results-content");

  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // Render Annotated Image
  document.getElementById("annotated-result-img").src = data.annotated_image;

  // Detection Status Badge
  const statusBadge = document.getElementById("detection-status-badge");
  if (data.face_detected) {
    statusBadge.innerHTML = `<i class="fa-solid fa-face-smile"></i> Face Detected`;
    statusBadge.style.color = "#34d399";
  } else {
    statusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Full Image Analyzed`;
    statusBadge.style.color = "#f59e0b";
  }

  // Gender Card Update
  const genderVal = document.getElementById("gender-val");
  const genderIcon = document.getElementById("gender-icon");
  const genderConfVal = document.getElementById("gender-conf-val");
  const genderProgress = document.getElementById("gender-progress-fill");

  genderVal.innerText = data.gender;
  genderConfVal.innerText = `${data.gender_confidence}%`;
  genderProgress.style.width = `${data.gender_confidence}%`;

  if (data.gender === "Female") {
    genderIcon.className = "fa-solid fa-venus metric-icon";
    genderIcon.style.color = "#e11d48";
    genderProgress.style.background = "linear-gradient(90deg, #e11d48, #db2777)";
  } else {
    genderIcon.className = "fa-solid fa-mars metric-icon";
    genderIcon.style.color = "#0284c7";
    genderProgress.style.background = "linear-gradient(90deg, #0284c7, #4f46e5)";
  }

  // Age Card Update
  animateCounter("age-val", parseInt(data.age));
  document.getElementById("age-range-val").innerText = data.age_range;
  document.getElementById("life-stage-val").innerText = data.age_group;
}

// Number Counter Animation
function animateCounter(elementId, targetVal) {
  const el = document.getElementById(elementId);
  let currentVal = 0;
  const step = Math.max(1, Math.floor(targetVal / 15));
  const timer = setInterval(() => {
    currentVal += step;
    if (currentVal >= targetVal) {
      currentVal = targetVal;
      clearInterval(timer);
    }
    el.innerText = currentVal;
  }, 25);
}
