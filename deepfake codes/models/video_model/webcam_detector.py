"""
webcam_detector.py  —  Debugged & Advanced
==========================================
BUGS FIXED:
  1. ✅ Model loaded at global scope (crashes if model missing)
  2. ✅ denoising on every frame (too slow, causes lag)
  3. ✅ Sharpening kernel too aggressive (over-sharpens)
  4. ✅ No FPS counter shown
  5. ✅ No recording capability
  6. ✅ Confidence threshold not configurable

NEW FEATURES:
  + Live FPS counter
  + Fake/Real history bar (last 30 frames)
  + Press 'r' to start/stop recording
  + Press 's' to save screenshot
  + Press 'q' to quit
  + Rolling average smoothing (avoids flickering)
  + Confidence threshold: only label if > 65%
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
from collections import deque

# ── Settings ─────────────────────────────────────────────
MODEL_PATH = r"C:\Users\sudhir chaturvedi\Desktop\deepfake dataset\deepfake codes\deepfake_final_model.pth"
DECISION_THRESHOLD = 0.595    # calibrated decision threshold
CONFIDENCE_THRESHOLD = 0.65   # only label if model is this confident
HISTORY_SIZE = 30             # rolling window for smoothing
PROCESS_EVERY = 3             # analyze every N frames (performance)

device = torch.device("cpu")
classes = ["Fake", "Real"]

# ── Load Model ───────────────────────────────────────────
print("Loading model...")
try:
    model = models.resnet18(weights=None)
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Auto-detect head architecture from saved weights
    if any(k.startswith("fc.1.") for k in state.keys()):
        # Trained with Sequential(Dropout, Linear) head
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 2)
        )
        print("  Head: Sequential(Dropout, Linear)")
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)
        print("  Head: Linear")
    model.load_state_dict(state, strict=True)  # strict=True catches mismatches
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    print("Make sure MODEL_PATH is correct")
    exit(1)

# ── Transform ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Face Detector ─────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.has_mp = False
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.has_mp = True
        except Exception:
            pass

    def get_face_crop(self, frame: np.ndarray) -> tuple:
        """Returns (crop, bbox_coords) or (None, None)"""
        if self.has_mp:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    results = face_detection.process(rgb)
                    if results.detections:
                        det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
                        bbox = det.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        box_w = min(w - x, int(bbox.width * w))
                        box_h = min(h - y, int(bbox.height * h))
                        
                        # Add 15% padding to match wider Haar cascade training crop style
                        padding = int(max(box_w, box_h) * 0.15)
                        x_min = max(0, x - padding)
                        y_min = max(0, y - padding)
                        x_max = min(w, x + box_w + padding)
                        y_max = min(h, y + box_h + padding)
                        
                        crop_w = x_max - x_min
                        crop_h = y_max - y_min
                        
                        # Make crop square
                        cx = x_min + crop_w // 2
                        cy = y_min + crop_h // 2
                        sz = max(crop_w, crop_h)
                        new_x = max(0, cx - sz // 2)
                        new_y = max(0, cy - sz // 2)
                        if new_x + sz > w:
                            new_x = max(0, w - sz)
                        if new_y + sz > h:
                            new_y = max(0, h - sz)
                        final_sz = min(sz, w - new_x, h - new_y)
                        
                        if final_sz >= 60:
                            return frame[new_y:new_y+final_sz, new_x:new_x+final_sz], (new_x, new_y, final_sz, final_sz)
            except Exception:
                pass

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                return frame[y:y+h, x:x+w], (x, y, w, h)
        except Exception:
            pass

        return None, None

face_detector = FaceDetector()

# ── Frame Enhancement (lighter version) ──────────────────
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Light enhancement: CLAHE only. No denoising (too slow)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Predict face crop ─────────────────────────────────────
def predict_face(face_bgr: np.ndarray):
    """Returns (fake_prob, real_prob) with 2-view TTA"""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil1 = Image.fromarray(rgb)
    pil2 = pil1.transpose(Image.FLIP_LEFT_RIGHT)
    
    t1 = transform(pil1).unsqueeze(0).to(device)
    t2 = transform(pil2).unsqueeze(0).to(device)
    t_batch = torch.cat([t1, t2], dim=0) # shape: (2, 3, 224, 224)
    
    with torch.no_grad():
        p = F.softmax(model(t_batch), dim=1)
        
    avg_p = p.mean(dim=0)
    fake_p = avg_p[0].item()
    real_p = avg_p[1].item()
    return fake_p, real_p


# ── Draw Overlay ─────────────────────────────────────────
def draw_overlay(frame, x, y, w, h, label, fake_p, real_p, conf, history):
    # Color
    if conf < CONFIDENCE_THRESHOLD:
        color = (0, 220, 220)  # cyan = uncertain
        display = f"? UNCERTAIN ({conf*100:.0f}%)"
    elif label == "Fake":
        color = (0, 0, 255)    # red = fake
        display = f"FAKE {fake_p*100:.1f}%"
    else:
        color = (0, 220, 100)  # green = real
        display = f"REAL {real_p*100:.1f}%"

    # Bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Corner markers (more polished than plain rectangle)
    sz = 18
    t = 3
    for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx*sz, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy*sz), color, t)

    # Label background
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y-th-14), (x+tw+12, y), color, -1)
    cv2.putText(frame, display, (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

    # History bar (last 30 predictions)
    bar_x, bar_y = x, y + h + 8
    for i, p in enumerate(list(history)):
        c = (0, 0, 200) if p >= DECISION_THRESHOLD else (0, 200, 80)
        cv2.rectangle(frame, (bar_x + i*6, bar_y), (bar_x + i*6+5, bar_y+8), c, -1)

    return frame


# ── Main Loop ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🟢 DeepShield Webcam Detection")
print("   Q = Quit  |  S = Screenshot  |  R = Record  |  E = Toggle enhance")

history: deque = deque([0.5] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
frame_count  = 0
fps_time     = time.time()
fps          = 0.0
recording    = False
enhance      = True
out_writer   = None
last_label   = "Analyzing..."
last_fake_p  = 0.5
last_real_p  = 0.5
last_conf    = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed")
        break

    frame_count += 1

    # FPS
    if frame_count % 15 == 0:
        fps = 15 / (time.time() - fps_time)
        fps_time = time.time()

    # Enhancement (for display only — prediction uses raw frame)
    display_frame = enhance_frame(frame) if enhance else frame.copy()

    # Face detection + prediction (every N frames)
    # IMPORTANT: detect & predict on RAW frame, not enhanced frame,
    # because the model was trained on un-enhanced images.
    if frame_count % PROCESS_EVERY == 0:
        face_crop, bbox = face_detector.get_face_crop(frame)

        if bbox is not None:
            x, y, w, h = bbox
            fake_p, real_p = predict_face(face_crop)
            history.append(fake_p)

            # Rolling average smoothing
            last_fake_p = float(np.mean(list(history)))
            last_real_p = 1.0 - last_fake_p
            last_label = classes[0] if last_fake_p >= DECISION_THRESHOLD else classes[1]
            last_conf = max(last_fake_p, last_real_p)

            display_frame = draw_overlay(
                display_frame, x, y, w, h,
                last_label, last_fake_p, last_real_p, last_conf, history
            )
    else:
        # Redraw last result on non-processed frames
        # Use raw frame for face detection consistency
        _, bbox = face_detector.get_face_crop(frame)
        if bbox is not None:
            x, y, w, h = bbox
            display_frame = draw_overlay(
                display_frame, x, y, w, h,
                last_label, last_fake_p, last_real_p, last_conf, history
            )

    # HUD
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Enhance: {'ON' if enhance else 'OFF'}",
        f"Recording: {'YES' if recording else 'NO'}",
        f"Frames: {frame_count}",
        f"Rolling avg: {np.mean(list(history))*100:.1f}% fake",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(display_frame, line, (10, 25 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,200,200), 1)

    # Recording
    if recording and out_writer:
        out_writer.write(display_frame)

    cv2.imshow("DeepShield — Real-Time Deepfake Detection  [Q=quit S=snap R=record E=enhance]",
               display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"deepshield_capture_{int(time.time())}.jpg"
        cv2.imwrite(fname, display_frame)
        print(f"📸 Saved: {fname}")
    elif key == ord('r'):
        if not recording:
            fname = f"deepshield_recording_{int(time.time())}.avi"
            h_vid, w_vid = display_frame.shape[:2]
            out_writer = cv2.VideoWriter(fname,
                cv2.VideoWriter_fourcc(*"XVID"), 20, (w_vid, h_vid))
            recording = True
            print(f"🔴 Recording: {fname}")
        else:
            recording = False
            if out_writer:
                out_writer.release()
                out_writer = None
            print("⏹ Recording stopped")
    elif key == ord('e'):
        enhance = not enhance
        print(f"Enhancement: {'ON' if enhance else 'OFF'}")

cap.release()
if out_writer:
    out_writer.release()
cv2.destroyAllWindows()
print("Webcam closed.")