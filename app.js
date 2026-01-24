
// ESM import of MediaPipe Tasks Vision (pinned version)
import { FilesetResolver, PoseLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const metrics = document.getElementById("metrics");
const startBtn = document.getElementById("start");
const cameraSel = document.getElementById("camera");

let landmarker = null;
let stream = null;
let running = false;

console.log("✅ app.js loaded (ESM)");

const IDX = {
  NOSE: 0, LEFT_EYE_INNER: 1, LEFT_EYE: 2, LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4, RIGHT_EYE: 5, RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  LEFT_MOUTH: 9, RIGHT_MOUTH: 10,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_PINKY: 17, RIGHT_PINKY: 18,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_THUMB: 21, RIGHT_THUMB: 22,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  LEFT_HEEL: 29, RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32
};

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

const IDEAL = {
  Left_Hip_Angle: 90,
  Left_Knee_Angle: 90,
  Left_Ankle_Angle: 90,
  Right_Hip_Angle: 90,
  Right_Knee_Angle: 90,
  Right_Ankle_Angle: 90
};

function resizeCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
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
    await listCameras();
    running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error("getUserMedia error:", err);
    metrics.textContent = "Camera error. Check permissions in the browser/OS settings.";
  }
}

function calculateAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = [b.x - a.x, b.y - a.y];
  const bc = [c.x - b.x, c.y - b.y];
  const dot = ab[0] * bc[0] + ab[1] * bc[1];
  const magAB = Math.hypot(ab[0], ab[1]);
  const magBC = Math.hypot(bc[0], bc[1]);
  if (magAB === 0 || magBC === 0) return null;
  let cos = dot / (magAB * magBC);
  cos = Math.max(-1, Math.min(1, cos));
  const angle = (Math.acos(cos) * 180) / Math.PI;
  return Math.round((180 - angle) * 100) / 100;
}

function getPoseAngles(landmarks) {
  if (!landmarks?.length) return null;

  const isValid = (lm) => lm && lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1;
  const safe = (i) => {
    const lm = landmarks[i];
    return isValid(lm) ? { x: lm.x, y: lm.y } : null;
  };
  const safeCalc = (a, b, c) => (a && b && c) ? calculateAngle(a, b, c) : "N/A";
  const G = { x: 0, y: 1 };

  const angles = {
    Left_Shoulder_Angle: safeCalc(safe(IDX.LEFT_HIP), safe(IDX.LEFT_SHOULDER), safe(IDX.LEFT_ELBOW)),
    Left_Elbow_Angle:    safeCalc(safe(IDX.LEFT_SHOULDER), safe(IDX.LEFT_ELBOW), safe(IDX.LEFT_WRIST)),
    Left_Hip_Angle:      safeCalc(safe(IDX.LEFT_SHOULDER), safe(IDX.LEFT_HIP), safe(IDX.LEFT_KNEE)),
    Left_Knee_Angle:     safeCalc(safe(IDX.LEFT_HIP), safe(IDX.LEFT_KNEE), safe(IDX.LEFT_ANKLE)),
    Left_Ankle_Angle:    safeCalc(safe(IDX.LEFT_KNEE), safe(IDX.LEFT_ANKLE), safe(IDX.LEFT_FOOT_INDEX)),

    Left_Shoulder_Ground_Angle: safeCalc(G, safe(IDX.LEFT_SHOULDER), safe(IDX.LEFT_HIP)),
    Left_Elbow_Ground_Angle:    safeCalc(G, safe(IDX.LEFT_ELBOW),    safe(IDX.LEFT_SHOULDER)),
    Left_Hip_Ground_Angle:      safeCalc(G, safe(IDX.LEFT_HIP),      safe(IDX.LEFT_KNEE)),
    Left_Knee_Ground_Angle:     safeCalc(G, safe(IDX.LEFT_KNEE),     safe(IDX.LEFT_ANKLE)),
    Left_Ankle_Ground_Angle:    safeCalc(G, safe(IDX.LEFT_ANKLE),    safe(IDX.LEFT_FOOT_INDEX)),

    Right_Shoulder_Angle:       safeCalc(safe(IDX.RIGHT_HIP), safe(IDX.RIGHT_SHOULDER), safe(IDX.RIGHT_ELBOW)),
    Right_Elbow_Angle:          safeCalc(safe(IDX.RIGHT_SHOULDER), safe(IDX.RIGHT_ELBOW), safe(IDX.RIGHT_WRIST)),
    Right_Hip_Angle:            safeCalc(safe(IDX.RIGHT_SHOULDER), safe(IDX.RIGHT_HIP),   safe(IDX.RIGHT_KNEE)),
    Right_Knee_Angle:           safeCalc(safe(IDX.RIGHT_HIP),      safe(IDX.RIGHT_KNEE),  safe(IDX.RIGHT_ANKLE)),
    Right_Ankle_Angle:          safeCalc(safe(IDX.RIGHT_KNEE),     safe(IDX.RIGHT_ANKLE), safe(IDX.RIGHT_FOOT_INDEX)),

    Right_Shoulder_Ground_Angle: safeCalc(G, safe(IDX.RIGHT_SHOULDER), safe(IDX.RIGHT_HIP)),
    Right_Elbow_Ground_Angle:    safeCalc(G, safe(IDX.RIGHT_ELBOW),    safe(IDX.RIGHT_SHOULDER)),
    Right_Hip_Ground_Angle:      safeCalc(G, safe(IDX.RIGHT_HIP),      safe(IDX.RIGHT_KNEE)),
    Right_Knee_Ground_Angle:     safeCalc(G, safe(IDX.RIGHT_KNEE),     safe(IDX.RIGHT_ANKLE)),
    Right_Ankle_Ground_Angle:    safeCalc(G, safe(IDX.RIGHT_ANKLE),    safe(IDX.RIGHT_FOOT_INDEX)),
  };

  return angles;
}

function colorClassFor(name, value) {
  if (!(name in IDEAL)) return "na";
  if (typeof value !== "number") return "na";
  const deviation = Math.abs(value - IDEAL[name]);
  if (deviation <= 5) return "good";
  if (deviation <= 10) return "warn";
  return "bad";
}

function renderMetrics(angles) {
  if (!angles) {
    metrics.textContent = "No person detected.";
    return;
  }
  const lines = [];
  for (const [name, ideal] of Object.entries(IDEAL)) {
    const val = angles[name];
    if (typeof val === "number") {
      const rounded = Math.round(val);
      const cls = colorClassFor(name, val);
      lines.push(`<span class="${cls}">${name.replaceAll('_',' ')}: ${rounded}°</span>  <span class="na">(Ideal: ${ideal}°)</span>`);
    } else {
      lines.push(`<span class="na">${name.replaceAll('_',' ')}: N/A (Ideal: ${ideal}°)</span>`);
    }
  }
  metrics.innerHTML = lines.join("<br>");
}

function drawPose(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks?.length) return;

  const W = canvas.width / (window.devicePixelRatio || 1);
  const H = canvas.height / (window.devicePixelRatio || 1);

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffee66";
  ctx.beginPath();
  for (const [a, b] of CUSTOM_CONNECTIONS) {
    const p = landmarks[a], q = landmarks[b];
    if (!p || !q) continue;
    const x1 = p.x * W, y1 = p.y * H;
    const x2 = q.x * W, y2 = q.y * H;
    ctx.moveTo(W - x1, y1);
    ctx.lineTo(W - x2, y2);
  }
  ctx.stroke();

  for (const i of RELEVANT_LANDMARKS) {
    const p = landmarks[i];
    if (!p) continue;
    const x = p.x * W, y = p.y * H;
    ctx.fillStyle = "#3df5b2";
    ctx.beginPath();
    ctx.arc(W - x, y, 4, 0, Math.PI * 2);
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

async function init() {
  try {
    // ✅ Version + host must match the import above
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

window.addEventListener("resize", resizeCanvasToVideo);
video.addEventListener("loadedmetadata", resizeCanvasToVideo);

init();
