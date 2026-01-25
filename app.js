console.log("Body Angle Calculator — v1.1-imtp-ranges");

// MediaPipe (pinned version)
import { FilesetResolver, PoseLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

const video   = document.getElementById("video");
const canvas  = document.getElementById("overlay");
const ctx     = canvas.getContext("2d");
const metrics = document.getElementById("metrics");
const startBtn= document.getElementById("start");
const cameraSel = document.getElementById("camera");
const mirrorToggle = document.getElementById("mirrorToggle");

let landmarker = null;
let stream = null;
let running = false;
let mirror = true; // default ON to match front camera use

console.log("✅ app.js loaded (ESM)");

// ---- Landmark indices (MediaPipe Pose) ----
const IDX = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32
};

// ---- Connections to draw ----
const RELEVANT_LANDMARKS = [
  IDX.LEFT_SHOULDER, IDX.RIGHT_SHOULDER,
  IDX.LEFT_ELBOW, IDX.RIGHT_ELBOW,
  IDX.LEFT_WRIST, IDX.RIGHT_WRIST,
  IDX.LEFT_HIP, IDX.RIGHT_HIP,
  IDX.LEFT_KNEE, IDX.RIGHT_KNEE,
  IDX.LEFT_ANKLE, IDX.RIGHT_ANKLE
];

const CUSTOM_CONNECTIONS = [
  [IDX.LEFT_SHOULDER, IDX.LEFT_ELBOW],
  [IDX.LEFT_ELBOW, IDX.LEFT_WRIST],
  [IDX.RIGHT_SHOULDER, IDX.RIGHT_ELBOW],
  [IDX.RIGHT_ELBOW, IDX.RIGHT_WRIST],
  [IDX.LEFT_HIP, IDX.LEFT_KNEE],
  [IDX.LEFT_KNEE, IDX.LEFT_ANKLE],
  [IDX.RIGHT_HIP, IDX.RIGHT_KNEE],
  [IDX.RIGHT_KNEE, IDX.RIGHT_ANKLE],
  [IDX.LEFT_SHOULDER, IDX.LEFT_HIP],
  [IDX.RIGHT_SHOULDER, IDX.RIGHT_HIP],
  [IDX.LEFT_SHOULDER, IDX.RIGHT_SHOULDER],
  [IDX.LEFT_HIP, IDX.RIGHT_HIP]
];

// ---- Ideal angle ranges (match your Python) ----
const IDEAL_RANGES = {
  Trunk_Angle:        [5, 10],
  Left_Hip_Angle:     [140, 150],
  Right_Hip_Angle:    [140, 150],
  Left_Knee_Angle:    [125, 145],
  Right_Knee_Angle:   [125, 145]
};

// ---- Helpers ----
function resizeCanvasToVideo() {
  const dpr = window.devicePixelRatio || 1;
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === "videoinput");
    cameraSel.innerHTML = "";
    for (const d of cams) {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${cameraSel.length + 1}`;
      cameraSel.appendChild(opt);
    }
  } catch (e) {
    console.warn("enumerateDevices failed (likely no permission yet). Will retry after start.", e);
  }
}

async function startCamera() {
  try {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: cameraSel.value ? { exact: cameraSel.value } : undefined,
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvasToVideo();
    applyMirror();
    await listCameras();
    running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error("getUserMedia error:", err);
    metrics.textContent = "Camera error. Check permissions in the browser/OS settings.";
  }
}

// ---- Your 2D angle (returns 180 - acos) ----
function calculateAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = [b.x - a.x, b.y - a.y];
  const bc = [c.x - b.x, c.y - b.y];
  const dot = ab[0]*bc[0] + ab[1]*bc[1];
  const magAB = Math.hypot(ab[0], ab[1]);
  const magBC = Math.hypot(bc[0], bc[1]);
  if (magAB === 0 || magBC === 0) return null;
  let cos = dot / (magAB * magBC);
  cos = Math.max(-1, Math.min(1, cos));
  const angle = (Math.acos(cos) * 180) / Math.PI;
  return Math.round((180 - angle) * 100) / 100;
}

function valid(lm) {
  return lm && lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1;
}
const safe = (lms, i) => (valid(lms?.[i]) ? { x: lms[i].x, y: lms[i].y } : null);

function midPoint(p1, p2) {
  if (!p1 || !p2) return null;
  return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
}

// Trunk angle vs vertical (0, -1), like your Python (no 180-… here)
function trunkAngleWithVertical(shoulder, hip) {
  if (!shoulder || !hip) return "N/A";
  const v = { x: shoulder.x - hip.x, y: shoulder.y - hip.y }; // hip -> shoulder
  const vertical = { x: 0, y: -1 }; // up in image coords
  const dot = v.x * vertical.x + v.y * vertical.y;
  const magV = Math.hypot(v.x, v.y);
  if (magV === 0) return "N/A";
  let cos = dot / magV; // /1 for vertical magnitude
  cos = Math.max(-1, Math.min(1, cos));
  const angle = Math.acos(cos) * 180 / Math.PI;
  return Math.round(angle * 100) / 100;
}

// Build all angles (match your Python)
function getPoseAngles(landmarks) {
  if (!landmarks?.length) return null;
  const L = (i) => safe(landmarks, i);

  // Midpoints
  const midShoulder = midPoint(L(IDX.LEFT_SHOULDER), L(IDX.RIGHT_SHOULDER));
  const midHip      = midPoint(L(IDX.LEFT_HIP),      L(IDX.RIGHT_HIP));

  const angles = {
    Trunk_Angle: trunkAngleWithVertical(midShoulder, midHip),

    Left_Shoulder_Angle:  calculateAngle(L(IDX.LEFT_HIP),  L(IDX.LEFT_SHOULDER), L(IDX.LEFT_ELBOW)) ?? "N/A",
    Left_Elbow_Angle:     calculateAngle(L(IDX.LEFT_SHOULDER), L(IDX.LEFT_ELBOW), L(IDX.LEFT_WRIST)) ?? "N/A",
    Left_Hip_Angle:       calculateAngle(L(IDX.LEFT_SHOULDER), L(IDX.LEFT_HIP),   L(IDX.LEFT_KNEE)) ?? "N/A",
    Left_Knee_Angle:      calculateAngle(L(IDX.LEFT_HIP),  L(IDX.LEFT_KNEE),  L(IDX.LEFT_ANKLE)) ?? "N/A",
    Left_Ankle_Angle:     calculateAngle(L(IDX.LEFT_KNEE), L(IDX.LEFT_ANKLE), L(IDX.LEFT_FOOT_INDEX)) ?? "N/A",

    Right_Shoulder_Angle: calculateAngle(L(IDX.RIGHT_HIP), L(IDX.RIGHT_SHOULDER), L(IDX.RIGHT_ELBOW)) ?? "N/A",
    Right_Elbow_Angle:    calculateAngle(L(IDX.RIGHT_SHOULDER), L(IDX.RIGHT_ELBOW), L(IDX.RIGHT_WRIST)) ?? "N/A",
    Right_Hip_Angle:      calculateAngle(L(IDX.RIGHT_SHOULDER), L(IDX.RIGHT_HIP),   L(IDX.RIGHT_KNEE)) ?? "N/A",
    Right_Knee_Angle:     calculateAngle(L(IDX.RIGHT_HIP), L(IDX.RIGHT_KNEE),  L(IDX.RIGHT_ANKLE)) ?? "N/A",
    Right_Ankle_Angle:    calculateAngle(L(IDX.RIGHT_KNEE), L(IDX.RIGHT_ANKLE), L(IDX.RIGHT_FOOT_INDEX)) ?? "N/A"
  };

  return angles;
}

// Range classification (match your Python’s deviation/percent_out logic)
function classifyRange(value, [min, max]) {
  if (typeof value !== "number") return "na";
  if (value >= min && value <= max) return "good";
  const width = Math.max(1e-6, max - min);
  const deviation = value < min ? (min - value) : (value - max);
  const percent_out = deviation / width;
  if (percent_out <= 0.2) return "warn"; // within 20% outside
  return "bad";
}

function renderMetrics(angles) {
  if (!angles) {
    metrics.textContent = "No person detected.";
    return;
  }
  const lines = [];
  const keysInOrder = Object.keys(IDEAL_RANGES);
  for (const name of keysInOrder) {
    const r = IDEAL_RANGES[name];
    const val = angles[name];
    if (typeof val === "number") {
      const cls = classifyRange(val, r);
      const rounded = Math.round(val);
      lines.push(
        `<span class="${cls}">${name.replaceAll('_',' ')}: ${rounded}°</span>  <span class="na">(Ideal: ${r[0]}–${r[1]}°)</span>`
      );
    } else {
      lines.push(
        `<span class="na">${name.replaceAll('_',' ')}: N/A (Ideal: ${r[0]}–${r[1]}°)</span>`
      );
    }
  }
  metrics.innerHTML = lines.join("<br>");
}

// Draw pose (respect mirror toggle)
function drawPose(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks?.length) return;

  const W = canvas.width / (window.devicePixelRatio || 1);
  const H = canvas.height / (window.devicePixelRatio || 1);
  const mapX = mirror ? (x) => W - x : (x) => x;

  // Lines
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#00f2ff";
  ctx.beginPath();
  for (const [a, b] of CUSTOM_CONNECTIONS) {
    const p = landmarks[a], q = landmarks[b];
    if (!p || !q) continue;
    const x1 = p.x * W, y1 = p.y * H;
    const x2 = q.x * W, y2 = q.y * H;
    ctx.moveTo(mapX(x1), y1);
    ctx.lineTo(mapX(x2), y2);
  }
  ctx.stroke();

  // Points
  ctx.fillStyle = "#03fc94";
  for (const i of RELEVANT_LANDMARKS) {
    const p = landmarks[i];
    if (!p) continue;
    const x = p.x * W, y = p.y * H;
    ctx.beginPath();
    ctx.arc(mapX(x), y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function loop() {
  if (!running || !landmarker || video.readyState < 2) {
    requestAnimationFrame(loop);
    return;
  }
  const ts = performance.now();
  const result = await landmarker.detectForVideo(video, ts);
  const lm = result?.landmarks?.[0] || null;

  drawPose(lm);
  const angles = lm ? getPoseAngles(lm) : null;
  renderMetrics(angles);

  requestAnimationFrame(loop);
}

// ---- Mirror handling ----
function applyMirror() {
  video.style.transform = mirror ? "scaleX(-1)" : "none";
}

mirrorToggle.addEventListener("change", () => {
  mirror = mirrorToggle.checked;
  applyMirror();
});

// ---- Init MediaPipe ----
async function init() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
    );

    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    startBtn.disabled = false;
    startBtn.textContent = "Start Workout";
    metrics.textContent = "Model loaded. Click Start to begin.";
    await listCameras();
  } catch (err) {
    console.error("❌ MediaPipe init failed:", err);
    metrics.textContent = "Failed to load model (WASM/CDN). Try a hard refresh or different network.";
  }
}

startBtn.addEventListener("click", () => {
  if (!landmarker) {
    metrics.textContent = "Model still loading…";
    return;
  }
  startCamera();
});

// Debounce utility
function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

const debouncedResize = debounce(resizeCanvasToVideo, 200);

window.addEventListener("resize", debouncedResize);
window.addEventListener("orientationchange", () => setTimeout(resizeCanvasToVideo, 200));
video.addEventListener("loadedmetadata", resizeCanvasToVideo);

init();
