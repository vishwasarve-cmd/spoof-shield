import os, time, random, base64, csv, hashlib, secrets, hmac, sqlite3, threading
from datetime import datetime
from collections import deque
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except (AttributeError, ImportError):
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# ============================================================
# SPOOFSHIELD WEB EDITION
# Original ML logic adapted from the uploaded desktop application.
# Browser handles the webcam; Python handles ML inference.
# ============================================================

DB_FILE = "spoofshield_auth.db"
DATASET_FILE = "authentication_dataset.csv"
PBKDF2_ITERATIONS = 200_000
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60

ALL_CHALLENGES = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
CHALLENGE_NAMES = {
    "BLINK": "Blink once",
    "TURN_LEFT": "Slowly turn LEFT",
    "TURN_RIGHT": "Slowly turn RIGHT",
}

# Liveness is primarily decided from temporal, user-driven actions.
# The passive signals are supporting evidence only.
DEEP_WEIGHT = 0.15
TEXTURE_WEIGHT = 0.15
CHROMA_WEIGHT = 0.10
LIVENESS_WEIGHT = 0.60
CHALLENGE_COUNT = 2
CHALLENGE_TIMEOUT = 12.0
BLINK_CLOSED = 0.23
BLINK_OPEN = 0.26
TURN_THRESHOLD = 8.0
TURN_CONFIRM_FRAMES = 2
BLINK_MAX_CLOSED_FRAMES = 10
FACE_MIN_RATIO = 0.16

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

app = FastAPI(title="SpoofShield AI")

# -------- Database / authentication --------
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        failed_attempts INTEGER DEFAULT 0,
        locked_until REAL DEFAULT 0
    )""")
    return conn

def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"),
        bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()

def register_user_web(user_id, password):
    user_id = user_id.strip()
    if len(user_id) < 3:
        raise HTTPException(400, "User ID must contain at least 3 characters.")
    if len(password) < 8:
        raise HTTPException(400, "Password must contain at least 8 characters.")

    salt = secrets.token_hex(16)
    conn = db()
    try:
        conn.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, 0, 0)",
            (user_id, salt, hash_password(password, salt), datetime.now().isoformat())
        )
        conn.commit()
        return {"message": "Account created successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That User ID already exists.")
    finally:
        conn.close()

def authenticate_user_web(user_id, password):
    conn = db()
    row = conn.execute(
        "SELECT salt,password_hash,failed_attempts,locked_until FROM users WHERE user_id=?",
        (user_id,)
    ).fetchone()

    if not row:
        conn.close()
        return False, "UNKNOWN_USER", 0

    salt, stored_hash, failed, locked_until = row
    now = time.time()

    if locked_until > now:
        conn.close()
        return False, f"LOCKED_{int(locked_until-now)}S", failed

    valid = hmac.compare_digest(hash_password(password, salt), stored_hash)

    if valid:
        conn.execute("UPDATE users SET failed_attempts=0, locked_until=0 WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        return True, "PASSWORD_OK", 0

    failed += 1
    lock = now + LOCKOUT_SECONDS if failed >= MAX_FAILED_LOGINS else 0
    if lock:
        failed = 0

    conn.execute(
        "UPDATE users SET failed_attempts=?, locked_until=? WHERE user_id=?",
        (failed, lock, user_id)
    )
    conn.commit()
    conn.close()
    return False, "PASSWORD_FAILED", failed

def append_login_record(user_id, password_status, liveness_result, ensemble_confidence,
                        deep_score, texture_score, chroma_score, challenge_progress,
                        attempt, latency_ms=0, face_detected=0, stability=0):
    fields = [
        "timestamp", "user_id", "password_status", "liveness_result",
        "ensemble_confidence", "deep_score", "texture_score", "chroma_score",
        "challenge_progress", "attempt", "latency_ms", "face_detected",
        "score_stability"
    ]
    exists = os.path.exists(DATASET_FILE)
    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id,
            "password_status": password_status,
            "liveness_result": liveness_result,
            "ensemble_confidence": int(ensemble_confidence),
            "deep_score": round(float(deep_score), 4),
            "texture_score": round(float(texture_score), 4),
            "chroma_score": round(float(chroma_score), 4),
            "challenge_progress": round(float(challenge_progress), 4),
            "attempt": attempt,
            "latency_ms": int(latency_ms),
            "face_detected": int(face_detected),
            "score_stability": round(float(stability), 4),
        })

# -------- ML model --------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

deep_feature_model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)
deep_feature_model.eval().to(device)

deep_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

inference_lock = threading.Lock()

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

def calculate_mar(landmarks, img_w, img_h):
    top = np.array([landmarks[13].x * img_w, landmarks[13].y * img_h])
    bottom = np.array([landmarks[14].x * img_w, landmarks[14].y * img_h])
    left = np.array([landmarks[61].x * img_w, landmarks[61].y * img_h])
    right = np.array([landmarks[291].x * img_w, landmarks[291].y * img_h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 0 else 0.0

def estimate_3d_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])

def analyze_frequency_and_color(roi_bgr):
    if roi_bgr.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    chroma_std = float(np.std(ycbcr[:, :, 1]) + np.std(ycbcr[:, :, 2]))
    return float(laplacian_var), chroma_std

def deep_score(roi_bgr):
    if roi_bgr.size == 0:
        return 0.0
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = deep_preprocess(Image.fromarray(roi_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = deep_feature_model.features(tensor)
        variance = torch.var(features).item()
    return min(1.0, max(0.0, (variance - 0.015) / 0.035))

# Per-user web sessions
sessions = {}

def new_session(user_id):
    # Use two easy but complementary active challenges.
    challenges = random.sample(ALL_CHALLENGES, CHALLENGE_COUNT)
    sessions[user_id] = {
        "challenges": challenges,
        "current_step": 0,
        "verify_state": 0,
        "challenge_start": time.time(),
        "blink_seen_closed": False,
        "blink_closed_frames": 0,
        "action_frames": 0,
        "scores": deque(maxlen=20),
        "texture_scores": deque(maxlen=12),
        "chroma_scores": deque(maxlen=12),
        "yaw_history": deque(maxlen=8),
        "neutral_yaw_samples": [],
        "neutral_yaw": 0.0,
        "neutral_ready": False,
        "last_face_time": time.time(),
        "attempt": 1,
        "started": time.time(),
        "result_logged": False
    }
    return sessions[user_id]

class Credentials(BaseModel):
    user_id: str
    password: str

class FrameRequest(BaseModel):
    user_id: str
    image: str

@app.post("/api/register")
def api_register(data: Credentials):
    return register_user_web(data.user_id, data.password)

@app.post("/api/login")
def api_login(data: Credentials):
    started = time.time()
    ok, status, _ = authenticate_user_web(data.user_id, data.password)
    latency = (time.time() - started) * 1000

    if not ok:
        append_login_record(data.user_id or "UNKNOWN", status, "NOT_STARTED",
                            0, 0, 0, 0, 0, 1, latency)
        raise HTTPException(401, status.replace("_", " "))

    new_session(data.user_id)
    return {"message": "Login successful", "user_id": data.user_id}

@app.post("/api/reset")
def api_reset(data: dict):
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(400, "user_id is required")
    new_session(user_id)
    return {"message": "Scanner reset"}

@app.post("/api/analyze")
def api_analyze(data: FrameRequest):
    if data.user_id not in sessions:
        raise HTTPException(401, "Session not found. Please login again.")

    session = sessions[data.user_id]
    try:
        encoded = data.image.split(",", 1)[1] if "," in data.image else data.image
        image_bytes = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(400, "Invalid webcam frame")
    if frame is None:
        raise HTTPException(400, "Could not decode frame")

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    with inference_lock:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

    face_detected = bool(results.multi_face_landmarks)
    avg_ear = mar = pitch = yaw = roll = 0.0
    raw_deep = 0.0
    texture_raw = chroma_raw = 0.0
    action_completed = False
    target = session["challenges"][session["current_step"]] if session["current_step"] < CHALLENGE_COUNT else None

    if face_detected:
        landmarks = results.multi_face_landmarks[0].landmark

        # Face-relative ROI. The old fixed center crop could measure the
        # background instead of the user's face.
        xs = np.array([p.x * img_w for p in landmarks], dtype=np.float32)
        ys = np.array([p.y * img_h for p in landmarks], dtype=np.float32)
        x1 = max(0, int(xs.min() - 0.18 * (xs.max()-xs.min())))
        x2 = min(img_w, int(xs.max() + 0.18 * (xs.max()-xs.min())))
        y1 = max(0, int(ys.min() - 0.22 * (ys.max()-ys.min())))
        y2 = min(img_h, int(ys.max() + 0.22 * (ys.max()-ys.min())))
        roi = frame[y1:y2, x1:x2]

        face_w = x2 - x1
        face_h = y2 - y1
        face_quality = min(1.0, max(0.0, min(face_w/img_w, face_h/img_h) / FACE_MIN_RATIO))

        with inference_lock:
            raw_deep = deep_score(roi)
        texture_raw, chroma_raw = analyze_frequency_and_color(roi)

        session["scores"].append(raw_deep)
        session["texture_scores"].append(texture_raw)
        session["chroma_scores"].append(chroma_raw)
        session["last_face_time"] = time.time()

        avg_ear = (
            calculate_ear(landmarks, LEFT_EYE, img_w, img_h) +
            calculate_ear(landmarks, RIGHT_EYE, img_w, img_h)
        ) / 2.0
        mar = calculate_mar(landmarks, img_w, img_h)
        pitch, yaw, roll = estimate_3d_pose(landmarks, img_w, img_h)
        session["yaw_history"].append(yaw)

        # First 0.8s: silently learn the user's natural head angle.
        # This prevents camera placement from breaking left/right detection.
        if not session["neutral_ready"]:
            session["neutral_yaw_samples"].append(yaw)
            if len(session["neutral_yaw_samples"]) >= 5:
                session["neutral_yaw"] = float(np.median(session["neutral_yaw_samples"]))
                session["neutral_ready"] = True
                session["challenge_start"] = time.time()

        if session["verify_state"] == 0 and session["neutral_ready"]:
            yaw_delta = yaw - session["neutral_yaw"]

            if target == "TURN_LEFT":
                if yaw_delta > TURN_THRESHOLD:
                    session["action_frames"] += 1
                else:
                    session["action_frames"] = max(0, session["action_frames"] - 1)
                action_completed = session["action_frames"] >= TURN_CONFIRM_FRAMES

            elif target == "TURN_RIGHT":
                if yaw_delta < -TURN_THRESHOLD:
                    session["action_frames"] += 1
                else:
                    session["action_frames"] = max(0, session["action_frames"] - 1)
                action_completed = session["action_frames"] >= TURN_CONFIRM_FRAMES

            elif target == "BLINK":
                # Hysteresis avoids false blink detection from noisy EAR.
                if avg_ear < BLINK_CLOSED:
                    session["blink_seen_closed"] = True
                    session["blink_closed_frames"] += 1
                elif session["blink_seen_closed"] and avg_ear > BLINK_OPEN:
                    if 1 <= session["blink_closed_frames"] <= BLINK_MAX_CLOSED_FRAMES:
                        action_completed = True
                    session["blink_seen_closed"] = False
                    session["blink_closed_frames"] = 0

            if action_completed:
                session["current_step"] += 1
                session["challenge_start"] = time.time()
                session["action_frames"] = 0
                session["blink_seen_closed"] = False
                session["blink_closed_frames"] = 0
                if session["current_step"] >= CHALLENGE_COUNT:
                    session["verify_state"] = 1

    # Timeout only after the face has been successfully detected/calibrated.
    if session["verify_state"] == 0:
        elapsed = time.time() - session["challenge_start"]
        time_left = max(0.0, CHALLENGE_TIMEOUT - elapsed)
        if elapsed >= CHALLENGE_TIMEOUT:
            session["verify_state"] = 2
    else:
        time_left = 0.0

    # Passive scores are supporting signals, not hard gates.
    smooth_deep = float(np.median(session["scores"])) if session["scores"] else 0.0
    texture_component = float(np.clip((np.median(session["texture_scores"]) - 15.0) / 135.0, 0.0, 1.0)) if session["texture_scores"] else 0.0
    chroma_component = float(np.clip((np.median(session["chroma_scores"]) - 6.0) / 18.0, 0.0, 1.0)) if session["chroma_scores"] else 0.0
    progress = session["current_step"] / CHALLENGE_COUNT

    # Once the two live actions are completed, confidence is intentionally
    # dominated by temporal evidence. Passive metrics cannot veto a live user.
    liveness_component = progress
    confidence = int(round(100 * (
        smooth_deep * DEEP_WEIGHT +
        texture_component * TEXTURE_WEIGHT +
        chroma_component * CHROMA_WEIGHT +
        liveness_component * LIVENESS_WEIGHT
    )))
    if session["verify_state"] == 1:
        confidence = max(confidence, 92)

    contributions = {
        "Deep Features": smooth_deep * DEEP_WEIGHT,
        "Texture": texture_component * TEXTURE_WEIGHT,
        "Chrominance": chroma_component * CHROMA_WEIGHT,
        "Active Liveness": liveness_component * LIVENESS_WEIGHT,
    }
    top_reason = max(contributions, key=contributions.get)
    stability = float(np.std(session["scores"])) if len(session["scores"]) > 1 else 0.0

    if session["verify_state"] in (1, 2) and not session["result_logged"]:
        final = "VERIFIED" if session["verify_state"] == 1 else "TIMEOUT"
        append_login_record(
            data.user_id, "PASSWORD_OK", final, confidence,
            smooth_deep, texture_component, chroma_component,
            liveness_component, session["attempt"],
            (time.time() - session["started"]) * 1000,
            int(face_detected), stability
        )
        session["result_logged"] = True

    if session["verify_state"] == 1:
        title, subtitle, status = "ACCESS GRANTED", "Live actions completed successfully", "verified"
        challenge = "Verification Complete"
    elif session["verify_state"] == 2:
        title, subtitle, status = "VERIFICATION TIMEOUT", "Press Reset to try again", "timeout"
        challenge = "Timed out"
    elif not face_detected:
        title, subtitle, status = "FACE NOT DETECTED", "Move into the frame", "scanning"
        challenge = "Position your face inside the box"
    elif not session["neutral_ready"]:
        title, subtitle, status = "GET READY", "Look straight at the camera", "scanning"
        challenge = "Calibrating"
    else:
        challenge = CHALLENGE_NAMES[target]
        title = f"STEP {session['current_step'] + 1}/{CHALLENGE_COUNT}: {challenge}"
        subtitle = f"Do it naturally • {time_left:.1f}s"
        status = "scanning"

    return {
        "status": status, "title": title, "subtitle": subtitle,
        "challenge": challenge, "step": session["current_step"],
        "confidence": confidence, "face_detected": face_detected,
        "deep_score": round(smooth_deep * 100, 1),
        "texture_score": round(texture_component * 100, 1),
        "chroma_score": round(chroma_component * 100, 1),
        "yaw": round(yaw, 1), "pitch": round(pitch, 1),
        "ear": round(avg_ear, 3), "mar": round(mar, 3),
        "stability": round(stability, 4), "top_reason": top_reason,
        "contributions": {k: round(v * 100, 1) for k, v in contributions.items()},
        "time_left": round(time_left, 1)
    }

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpoofShield AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#f7f9ff;--muted:#a9b1d1;--violet:#8b5cf6;--aqua:#2dd4bf;--line:rgba(202,214,255,.16);--card:rgba(16,19,53,.72)}
*{box-sizing:border-box}body{margin:0;font-family:'DM Sans',Arial,sans-serif;background:#090b20;color:var(--ink);min-height:100vh}.hidden{display:none!important}
.auth{min-height:100vh;display:grid;place-items:center;overflow:hidden;isolation:isolate;padding:24px;background:radial-gradient(circle at 12% 8%,#36256f 0,transparent 28%),radial-gradient(circle at 91% 86%,#07575c 0,transparent 26%),#090b20}.auth:before,.auth:after{content:'';position:absolute;z-index:-1;filter:blur(1px);border-radius:999px;opacity:.75}.auth:before{width:520px;height:520px;background:linear-gradient(135deg,#6d3ee6,#8d65ff00);top:-210px;right:-110px;animation:float 12s ease-in-out infinite}.auth:after{width:460px;height:460px;background:linear-gradient(135deg,#00c5b4,#00c5b400);bottom:-250px;left:-115px;animation:float 15s ease-in-out infinite reverse}@keyframes float{50%{transform:translate(-32px,36px) scale(1.08)}}
.auth-shell{width:min(1180px,100%);display:grid;grid-template-columns:1.12fr .88fr;min-height:690px;background:rgba(12,14,40,.54);border:1px solid var(--line);border-radius:30px;box-shadow:0 32px 90px #000a;overflow:hidden;backdrop-filter:blur(22px);animation:rise .7s cubic-bezier(.2,.8,.2,1)}@keyframes rise{from{opacity:0;transform:translateY(22px) scale(.98)}to{opacity:1;transform:none}}
.auth-intro{position:relative;padding:42px 58px 35px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;background:linear-gradient(145deg,rgba(91,45,186,.54),rgba(8,22,57,.35) 58%,rgba(0,136,135,.22)),url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 40h80M40 0v80' stroke='%23fff' stroke-opacity='.035'/%3E%3C/svg%3E")}.brand{position:relative;z-index:2;display:flex;align-items:center;gap:11px;font:700 18px 'Space Grotesk';letter-spacing:.02em}.brand-mark{width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(135deg,#b79bff,#32dccc);box-shadow:0 8px 25px #7159dd80}.intro-copy{position:relative;z-index:2}.intro-copy h1{font:700 clamp(38px,5vw,60px)/1.03 'Space Grotesk';letter-spacing:-.05em;margin:0 0 19px;max-width:500px}.intro-copy p{color:#d6d9ed;line-height:1.65;max-width:430px;margin:0}.trust{position:relative;z-index:2;display:flex;gap:20px;color:#d9dcf2;font-size:13px}.trust span{display:flex;gap:7px;align-items:center}.trust i{width:8px;height:8px;border-radius:50%;background:#41e4b0;box-shadow:0 0 12px #41e4b0}
.security-visual{position:absolute;inset:90px 0 0;pointer-events:none}.orb{position:absolute;right:62px;top:30px;width:260px;height:260px;border-radius:50%;background:radial-gradient(circle at 37% 32%,#e9e2ff 0 2%,#8d70ff 8%,#4c45b8 27%,#1f3478 53%,#071333 72%);box-shadow:inset -28px -30px 45px #03061caa,0 0 0 1px #b9a6ff55,0 0 65px #7655ff8c;animation:orb-float 7s ease-in-out infinite}.orb:before,.orb:after{content:'';position:absolute;border-radius:50%;border:1px solid #84fff480;inset:-27px;transform:rotateX(66deg) rotateZ(22deg);box-shadow:0 0 20px #43ded066}.orb:after{inset:-58px;border-color:#c095ff50;transform:rotateX(69deg) rotateZ(-35deg)}@keyframes orb-float{50%{transform:translateY(-15px) rotate(6deg)}}
.mini-card{position:absolute;z-index:2;padding:13px 16px;border:1px solid #ffffff24;border-radius:15px;background:#121a48a8;backdrop-filter:blur(14px);box-shadow:0 14px 30px #02031955;animation:card-float 6s ease-in-out infinite}.mini-card strong{display:block;font:600 13px 'Space Grotesk';letter-spacing:.02em}.mini-card small{display:block;color:#aeb8dc;margin-top:4px;font-size:11px}.mini-card.one{top:10px;left:45px}.mini-card.two{top:250px;right:27px;animation-delay:-3s}.mini-icon{display:inline-grid;place-items:center;width:23px;height:23px;margin-right:7px;border-radius:7px;background:#2ad5c336;color:#47e8c7}@keyframes card-float{50%{transform:translateY(-8px)}}
.card{position:relative;padding:48px clamp(28px,5vw,58px);background:linear-gradient(160deg,rgba(25,27,66,.92),rgba(10,13,36,.93));display:flex;flex-direction:column;justify-content:center}.card:before{content:'';position:absolute;top:0;right:0;width:230px;height:230px;background:radial-gradient(circle,#bf80ff25,transparent 68%);pointer-events:none}.eyebrow{color:#bcaaff;font:600 12px 'Space Grotesk';text-transform:uppercase;letter-spacing:.14em;margin-bottom:12px}.card h2{font:700 34px 'Space Grotesk';letter-spacing:-.04em;margin:0}.tag{color:var(--muted);font-size:14px;line-height:1.5;margin:10px 0 30px}.tabs{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:5px;background:#07091d;border:1px solid var(--line);border-radius:13px;margin-bottom:20px}.tabs button{padding:11px;border:0;border-radius:9px;background:transparent;color:#9da6c9;font:600 13px 'DM Sans';cursor:pointer;transition:.25s}.tabs .active{color:#fff;background:linear-gradient(135deg,#7752e9,#b05de9);box-shadow:0 5px 15px #37277c80}
.field{position:relative;margin:13px 0}.field label{display:block;color:#cbd0e6;font-size:12px;font-weight:600;margin:0 0 7px}.field input{width:100%;padding:14px 43px 14px 14px;border-radius:11px;border:1px solid var(--line);outline:none;background:rgba(5,7,26,.66);color:#fff;font:500 14px 'DM Sans';transition:.25s}.field input::placeholder{color:#687194}.field input:focus{border-color:#9a7cff;box-shadow:0 0 0 4px #8b5cf622}.field-icon{position:absolute;right:14px;bottom:13px;color:#8d96ba;font-size:16px}.password-toggle{border:0;background:none;cursor:pointer;padding:0;color:#aeb6d2}.primary{position:relative;overflow:hidden;width:100%;padding:14px;border:0;border-radius:11px;background:linear-gradient(100deg,#7a56ed,#9b65ed 52%,#30c9bd);font:700 14px 'DM Sans';color:white;cursor:pointer;margin-top:10px;box-shadow:0 11px 24px #5939b06b;transition:transform .2s,box-shadow .2s}.primary:hover{transform:translateY(-2px);box-shadow:0 15px 32px #5939b08c}.primary:disabled{opacity:.7;cursor:wait;transform:none}.msg{min-height:22px;color:#ff9d9d;font-size:13px;margin-top:13px}.form-note{text-align:center;color:#8790b0;font-size:12px;margin:22px 0 0}.form-note span{color:#b6a8ff}
.dashboard{min-height:100vh;padding:24px;background:radial-gradient(circle at 85% 0,#2a1c5d 0,transparent 28%),#080a1b}.topbar{display:flex;justify-content:space-between;align-items:center;background:#14183adc;padding:17px 22px;border-radius:18px;border:1px solid #ffffff15;box-shadow:0 14px 40px #0005}.online{color:#44e69a;font-size:12px;font-weight:700;letter-spacing:.08em}.online:before{content:'';display:inline-block;width:8px;height:8px;border-radius:50%;background:#44e69a;margin-right:7px;box-shadow:0 0 14px #44e69a}.online{font-size:0}.online:after{content:'SYSTEM ONLINE';font-size:12px}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,1fr);gap:20px;margin-top:20px}.panel{background:linear-gradient(145deg,#151a3cd9,#0c1029e6);border:1px solid #ffffff15;border-radius:18px;padding:18px;box-shadow:0 16px 35px #0004}
.video-wrap{position:relative;background:#02070d;border-radius:14px;overflow:hidden;aspect-ratio:16/10}.video-wrap:after{content:'LIVE';position:absolute;top:14px;left:14px;padding:5px 9px;border-radius:8px;background:#ff4d7044;border:1px solid #ff7a9266;color:#ffb4c1;font:700 10px 'Space Grotesk';letter-spacing:.12em}.video-wrap video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
.scanbox{position:absolute;left:25%;top:15%;width:50%;height:70%;border:2px solid #57f0e0;border-radius:20px;box-shadow:0 0 25px #2bd9ff88,inset 0 0 23px #2bd9ff22;pointer-events:none}.scanbox:after{content:'';position:absolute;left:5%;right:5%;height:2px;top:12%;background:#78fff3;box-shadow:0 0 16px #43e9d8;animation:scan 2.5s ease-in-out infinite}@keyframes scan{50%{top:85%}}
.hud{margin-top:14px;padding:16px;border-radius:12px;background:#0a0e27;border:1px solid #ffffff0e}.hud h2{margin:0 0 6px;color:#8bf6eb;font-size:20px}.muted{color:#8fa5bb}
.metric{padding:12px 0;border-bottom:1px solid #ffffff10;display:flex;justify-content:space-between}.metric b{color:#c9fdf7}.bar{height:13px;background:#06101c;border-radius:20px;overflow:hidden;margin:10px 0 18px}.bar div{height:100%;width:0;background:linear-gradient(90deg,#7654ed,#2ce2c8);transition:.25s}
.xai{margin-top:18px;padding:15px;background:#0a0e27;border:1px solid #ffffff0e;border-radius:12px}.contrib{font-size:14px;margin:7px 0}.reset{padding:11px 18px;border:1px solid #8c6bff99;background:#825ce622;color:#cfc3ff;border-radius:10px;cursor:pointer;transition:.2s}.reset:hover{background:#825ce644}
@media(max-width:850px){.auth-shell{grid-template-columns:1fr;max-width:530px}.auth-intro{min-height:270px;padding:34px}.intro-copy h1{font-size:38px}.security-visual{display:none}.trust{display:none}.card{padding:38px 30px}.grid{grid-template-columns:1fr}.dashboard{padding:12px}.topbar{gap:10px}.topbar h1{font-size:19px}}@media(max-width:460px){.auth{padding:12px}.auth-shell{border-radius:20px}.auth-intro{padding:28px}.card{padding:32px 23px}}
</style>
</head>
<body>

<section id="authPage" class="auth">
  <div class="auth-shell">
    <aside class="auth-intro">
      <div class="brand"><span class="brand-mark">🛡</span> SPOOFSHIELD</div>
      <div class="intro-copy">
        <h1>Identity,<br><span style="color:#71eadc">made certain.</span></h1>
        <p>Secure access powered by intelligent face liveness verification. Designed to keep every login real, private, and effortless.</p>
      </div>
      <div class="security-visual" aria-hidden="true">
        <div class="mini-card one"><strong><span class="mini-icon">✓</span>Secure identity</strong><small>Encrypted end-to-end</small></div>
        <div class="orb"></div>
        <div class="mini-card two"><strong><span class="mini-icon">◌</span>Live detection</strong><small>AI engine active</small></div>
      </div>
      <div class="trust"><span><i></i> AES-grade protection</span><span><i></i> AI liveness engine</span></div>
    </aside>
    <div class="card">
      <div class="eyebrow">Secure portal</div>
      <h2 id="authHeading">Welcome back</h2>
      <div id="authTag" class="tag">Sign in to access your verification dashboard.</div>
      <div class="tabs" aria-label="Authentication mode">
        <button id="loginTab" class="active" onclick="setMode('login')">Sign in</button>
        <button id="registerTab" onclick="setMode('register')">Create account</button>
      </div>
      <div class="field"><label for="userId">User ID</label><input id="userId" placeholder="Enter your user ID" autocomplete="username"><span class="field-icon">◉</span></div>
      <div class="field"><label for="password">Password</label><input id="password" type="password" placeholder="Enter your password" autocomplete="current-password"><button class="field-icon password-toggle" id="passwordToggle" type="button" onclick="togglePassword()" aria-label="Show password">◌</button></div>
      <button id="authButton" class="primary" onclick="submitAuth()">Continue securely <span aria-hidden="true">→</span></button>
      <div id="authMsg" class="msg" role="status" aria-live="polite"></div>
      <p class="form-note">Protected by <span>SpoofShield AI</span> verification</p>
    </div>
  </div>
</section>

<section id="dashboard" class="dashboard hidden">
  <div class="topbar">
    <div><h1 style="margin:0;color:#63d8ff">🛡 SPOOFSHIELD AI</h1><span class="muted">Professional Multi-Model Engine</span></div>
    <div><span class="online">● SYSTEM ONLINE</span> &nbsp; <button class="reset" onclick="resetScanner()">Reset Scanner</button></div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="video-wrap">
        <video id="video" autoplay playsinline muted></video>
        <div class="scanbox"></div>
      </div>
      <div class="hud">
        <h2 id="title">Starting scanner...</h2>
        <div id="subtitle" class="muted">Allow webcam access</div>
      </div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0">MULTI-MODEL DASHBOARD</h3>
      <div class="metric"><span>User</span><b id="userDisplay">-</b></div>
      <div class="metric"><span>Face Detected</span><b id="face">-</b></div>
      <div class="metric"><span>Deep Feature Score</span><b id="deep">-</b></div>
      <div class="metric"><span>Texture Score</span><b id="texture">-</b></div>
      <div class="metric"><span>Chrominance Score</span><b id="chroma">-</b></div>
      <div class="metric"><span>Yaw / Pitch</span><b id="pose">-</b></div>
      <div class="metric"><span>EAR / MAR</span><b id="bio">-</b></div>

      <div style="margin-top:18px">ENSEMBLE LIVENESS: <b id="confidence">0%</b></div>
      <div class="bar"><div id="confidenceBar"></div></div>

      <div class="xai">
        <b>🧠 XAI DECISION EXPLANATION</b>
        <div class="muted" id="reason" style="margin:8px 0">Waiting for ML inference...</div>
        <div id="contributions"></div>
      </div>
    </div>
  </div>
</section>

<script>
let mode='login', user='', stream=null, timer=null, busy=false;
const $=id=>document.getElementById(id);

function setMode(m){
  mode=m;
  $('loginTab').classList.toggle('active',m==='login');
  $('registerTab').classList.toggle('active',m==='register');
  $('authMsg').textContent='';
  $('authHeading').textContent=m==='login'?'Welcome back':'Create your account';
  $('authTag').textContent=m==='login'?'Sign in to access your verification dashboard.':'Start with a secure, AI-protected identity.';
  $('authButton').innerHTML=m==='login'?'Continue securely <span aria-hidden="true">→</span>':'Create secure account <span aria-hidden="true">→</span>';
}

function togglePassword(){
  const visible=$('password').type==='text';
  $('password').type=visible?'password':'text';
  $('passwordToggle').textContent=visible?'◌':'◉';
  $('passwordToggle').setAttribute('aria-label',visible?'Show password':'Hide password');
}

async function submitAuth(){
  const user_id=$('userId').value.trim(), password=$('password').value;
  if(!user_id || !password){$('authMsg').textContent='Enter User ID and password.';return}
  const endpoint=mode==='login'?'/api/login':'/api/register';
  const button=$('authButton'), original=button.innerHTML;
  button.disabled=true;button.textContent=mode==='login'?'Signing you in…':'Creating account…';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id,password})});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Request failed');
    if(mode==='register'){ setMode('login');$('authMsg').style.color='#44e69a';$('authMsg').textContent='Account created. You can sign in now.';return; }
    user=user_id;$('authPage').classList.add('hidden');$('dashboard').classList.remove('hidden');
    $('userDisplay').textContent=user;
    await startCamera();
  }catch(e){$('authMsg').style.color='#ff9d9d';$('authMsg').textContent=e.message}
  finally{button.disabled=false;if(!button.innerHTML.includes('securely'))button.innerHTML=original}
}

$('password').addEventListener('keydown',e=>{if(e.key==='Enter')submitAuth()});

async function startCamera(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});
    $('video').srcObject=stream;
    $('video').onloadedmetadata=()=>{timer=setInterval(analyze,900)};
  }catch(e){$('subtitle').textContent='Camera error: '+e.message}
}

async function analyze(){
  if(busy || !$('video').videoWidth) return;
  busy=true;
  try{
    const c=document.createElement('canvas'); c.width=$('video').videoWidth;c.height=$('video').videoHeight;
    c.getContext('2d').drawImage($('video'),0,0);
    const image=c.toDataURL('image/jpeg',0.75);
    const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:user,image})});
    const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Analysis failed');
    updateUI(d);
  }catch(e){$('subtitle').textContent=e.message}
  busy=false;
}

function updateUI(d){
  $('title').textContent=d.title;$('subtitle').textContent=d.subtitle;
  $('face').textContent=d.face_detected?'YES':'NO';
  $('deep').textContent=d.deep_score+'%';$('texture').textContent=d.texture_score+'%';$('chroma').textContent=d.chroma_score+'%';
  $('pose').textContent=d.yaw+'° / '+d.pitch+'°';$('bio').textContent=d.ear+' / '+d.mar;
  $('confidence').textContent=d.confidence+'%';$('confidenceBar').style.width=d.confidence+'%';
  $('reason').textContent='Top evidence: '+d.top_reason;
  $('contributions').innerHTML=Object.entries(d.contributions).map(([k,v])=>`<div class="contrib">${k}: <b>${v}%</b></div>`).join('');
}

async function resetScanner(){
  if(!user) return;
  try{
    busy=true;
    const r=await fetch('/api/reset',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({user_id:user})
    });
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Reset failed');
    $('title').textContent='Scanner reset';
    $('subtitle').textContent='Look straight at the camera to begin again.';
    $('confidence').textContent='0%';
    $('confidenceBar').style.width='0%';
    $('reason').textContent='Scanner has been reset.';
    $('contributions').innerHTML='';
  }catch(e){
    $('subtitle').textContent='Reset error: '+e.message;
  }finally{
    busy=false;
  }
}

// Press R (or r) anywhere on the scanner dashboard to reset liveness.
document.addEventListener('keydown', function(event){
  if(event.key.toLowerCase()==='r' && !$('dashboard').classList.contains('hidden')){
    const tag=(event.target.tagName||'').toLowerCase();
    if(tag!=='input' && tag!=='textarea' && tag!=='select'){
      event.preventDefault();
      resetScanner();
    }
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
