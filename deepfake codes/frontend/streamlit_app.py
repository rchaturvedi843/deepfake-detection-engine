"""
streamlit_app.py  —  DeepShield v3.0
======================================
OPTIMIZED FOR FAST LOADING WITH MEDIAPIPE FIX & ENSEMBLE SUPPORT
"""

import sys
import os
import tempfile
import time

# ── Proper path management ───────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import plotly.graph_objects as go

# ── Optional imports with better error handling ───────────────────────────────────
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        HAS_MEDIAPIPE = True
    else:
        print("⚠️ MediaPipe installed but solutions module not found")
except ImportError:
    print("⚠️ MediaPipe not installed")
except Exception as e:
    print(f"⚠️ MediaPipe error: {e}")

HAS_REPORTLAB = False
try:
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except ImportError:
    print("⚠️ ReportLab not installed")

# ── Video model import with fallback ─────────────────────
HAS_VIDEO_PREDICTOR = False
try:
    from models.video_model.video_predictor import predict_video, get_predictor
    HAS_VIDEO_PREDICTOR = True
except ImportError:
    print("⚠️ Video predictor module not found")

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG (OPTIMIZED FOR SPEED)
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepShield — AI Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Cinematic Scan-Line Dark Theme
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #060a10;
    color: #c8d8e8;
}
.main { background: #060a10; }
.block-container { padding-top: 0.75rem; max-width: 1400px; }

.main::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,200,0.012) 2px, rgba(0,255,200,0.012) 4px);
    pointer-events: none;
    z-index: 0;
}

.ds-header {
    background: linear-gradient(135deg, #060e1a 0%, #0a1628 60%, #060e1a 100%);
    border: 1px solid #0d3060;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.ds-header::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffe0, #0066ff, transparent);
}
.ds-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0;
}
.ds-title span { color: #00ffe0; }
.ds-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: #3a7a9a;
    font-size: 0.75rem;
    letter-spacing: 2px;
    margin-top: 0.3rem;
}

.result-fake {
    background: linear-gradient(135deg, #1a0505, #2a0808);
    border: 1px solid #ff2244;
    border-left: 4px solid #ff2244;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 0 40px rgba(255,34,68,0.12), inset 0 0 20px rgba(255,34,68,0.04);
}
.result-real {
    background: linear-gradient(135deg, #041208, #071a0e);
    border: 1px solid #00cc88;
    border-left: 4px solid #00cc88;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 0 40px rgba(0,204,136,0.12), inset 0 0 20px rgba(0,204,136,0.04);
}
.result-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 5px;
    margin: 0;
}
.result-fake .result-label { color: #ff2244; }
.result-real .result-label { color: #00cc88; }
.result-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #5a8aaa;
    margin-top: 0.4rem;
    letter-spacing: 1px;
}

.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.75rem 0; }
.metric-tile {
    background: #080f1a;
    border: 1px solid #0d2a40;
    border-radius: 4px;
    padding: 0.8rem 1.1rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
    position: relative;
}
.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffe0, transparent);
}
.metric-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.5rem;
    font-weight: 400;
    color: #00ffe0;
    margin: 0;
}
.metric-lbl {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.7rem;
    color: #3a6a8a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

.sec-head {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #4a9aca;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #0d2a40;
    padding-bottom: 0.4rem;
    margin: 1.25rem 0 0.75rem;
}

.warn-box {
    background: #0d0a00;
    border-left: 3px solid #ffaa00;
    padding: 0.6rem 0.9rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #ffaa00;
    margin: 0.5rem 0;
}

.badge {
    display: inline-block;
    border-radius: 2px;
    font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace;
    padding: 2px 8px;
    letter-spacing: 1px;
    font-weight: 500;
}
.badge-high   { background: #002a1a; color: #00cc88; border: 1px solid #00cc88; }
.badge-medium { background: #1a1000; color: #ffaa00; border: 1px solid #ffaa00; }
.badge-low    { background: #1a0500; color: #ff6644; border: 1px solid #ff6644; }

[data-testid="stSidebar"] { background: #04080f !important; border-right: 1px solid #0d2a40; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { font-family: 'Exo 2', sans-serif; }

.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #0d2a40; }
.stTabs [data-baseweb="tab"] { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; letter-spacing: 2px; color: #3a6a8a; }
.stTabs [aria-selected="true"] { color: #00ffe0 !important; border-bottom-color: #00ffe0 !important; }

.stButton > button {
    background: transparent;
    border: 1px solid #0d3060;
    border-radius: 2px;
    color: #4a9aca;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.2s;
}
.stButton > button:hover { border-color: #00ffe0; color: #00ffe0; box-shadow: 0 0 12px rgba(0,255,224,0.2); }

.stProgress > div > div { background: linear-gradient(90deg, #0066ff, #00ffe0); }

[data-testid="stFileUploader"] {
    background: #080f1a !important;
    border: 1px dashed #0d2a40 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SETTINGS & PATHS
# ════════════════════════════════════════════════════════════
device = torch.device("cpu")
CALIBRATED_IMAGE_THRESHOLD = 0.595
CALIBRATED_ENSEMBLE = (
    ("deepfake_final_model.pth", 0.9),
    ("deepfake_strong_model.pth", 0.1),
)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ════════════════════════════════════════════════════════════
#  MODEL LOAD (OPTIMIZED WITH CACHING)
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_image_model(model_name: str):
    """Load model once and cache it"""
    try:
        model_path = os.path.join(PROJECT_ROOT, model_name)
        m = models.resnet18(weights=None)
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
        
        # Load on CPU
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        if any(k.startswith("fc.1.") for k in state.keys()):
            m.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(m.fc.in_features, 2)
            )
        else:
            m.fc = nn.Linear(m.fc.in_features, 2)

        m.load_state_dict(state, strict=True)
        m.eval()
        
        # Disable gradients for faster inference
        torch.set_grad_enabled(False)
        
        return m
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_image_ensemble():
    """Load the calibrated high-accuracy image ensemble."""
    loaded = []
    for model_name, weight in CALIBRATED_ENSEMBLE:
        loaded_model = load_image_model(model_name)
        if loaded_model is None:
            return None
        loaded.append((loaded_model, weight))
    return tuple(loaded)

# ════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
def detect_face_cv(image: Image.Image) -> Image.Image:
    """Detect face using Haar Cascade"""
    img = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return image
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return Image.fromarray(img[y:y+h, x:x+w])


def predict_image(image: Image.Image, model):
    """Predict image - Returns (result, confidence%, fake_prob%, real_prob%)"""
    if model is None:
        return None, 0, 0, 0
    
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        if isinstance(model, tuple):
            probs = sum(
                weight * torch.softmax(loaded_model(tensor), dim=1)
                for loaded_model, weight in model
            )
        else:
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
    
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    result = "Fake" if fake_prob >= CALIBRATED_IMAGE_THRESHOLD else "Real"
    confidence = max(fake_prob, real_prob)
    return (result, confidence * 100, fake_prob * 100, real_prob * 100)


def manipulation_heatmap(image: Image.Image) -> np.ndarray:
    """Generate manipulation heatmap"""
    img = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)


def generate_gradcam(image: Image.Image, model) -> np.ndarray:
    """Generate GradCAM visualization - FIXED FOR ENSEMBLE"""
    if model is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # If model is an ensemble (tuple), use the first model for GradCAM
    if isinstance(model, tuple):
        main_model = model[0][0]
    else:
        main_model = model
    
    img = image.convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    out = main_model(tensor)
    main_model.zero_grad()
    out[0, out.argmax()].backward()
    grad = tensor.grad.data.numpy()[0]
    hm = np.mean(grad, axis=0)
    hm = np.maximum(hm, 0)
    if hm.max() > 0:
        hm /= hm.max()
    hm = cv2.resize(hm, (224, 224))
    hm = np.uint8(255 * hm)
    return cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO)


def fft_analysis(image: Image.Image) -> np.ndarray:
    """Perform FFT analysis"""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    mag = 20 * np.log(np.abs(np.fft.fftshift(f)) + 1)
    mag = ((mag - mag.min()) / (mag.max() - mag.min()) * 255).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)


def gan_artifact_score(image: Image.Image) -> float:
    """Calculate GAN artifact score"""
    img = np.array(image.convert("RGB")).astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return round(float(np.mean(abs(r-g)) + np.mean(abs(g-b)) + np.mean(abs(r-b))), 3)


def face_consistency_score(image: Image.Image) -> float:
    """Calculate face consistency score"""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    h, w = gray.shape
    return round(float(np.sum(edges) / (h * w)), 4)


def face_landmarks_img(image: Image.Image) -> np.ndarray:
    """Detect face landmarks with better error handling"""
    img = np.array(image.convert("RGB"))
    if not HAS_MEDIAPIPE:
        return img
    
    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as fm:
            res = fm.process(img)
        if not res.multi_face_landmarks:
            return img
        out = img.copy()
        h, w = out.shape[:2]
        for lms in res.multi_face_landmarks:
            for lm in lms.landmark:
                cv2.circle(out, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 180), -1)
        return out
    except Exception as e:
        print(f"Face landmarks error: {e}")
        return img


def eye_blink_status(image: Image.Image) -> str:
    """Check eye blink status with better error handling"""
    if not HAS_MEDIAPIPE:
        return "MediaPipe not installed"
    
    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as fm:
            res = fm.process(np.array(image.convert("RGB")))
        if not res.multi_face_landmarks:
            return "No face detected"
        lm = res.multi_face_landmarks[0].landmark
        dist = abs(lm[159].y - lm[145].y)
        return "👁️ Eye Open" if dist >= 0.01 else "😑 Eye Closed"
    except Exception as e:
        print(f"Eye blink error: {e}")
        return "Error detecting eyes"


def make_gauge(fake_prob: float) -> go.Figure:
    """Create gauge chart"""
    color = "#ff2244" if fake_prob > 50 else "#00cc88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fake_prob,
        number={"suffix": "%", "font": {"size": 26, "color": color, "family": "Share Tech Mono"}},
        title={"text": "FAKE PROBABILITY", "font": {"size": 11, "color": "#3a6a8a", "family": "Rajdhani"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#0d2a40", "tickfont": {"color": "#3a6a8a", "size": 9}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#060a10",
            "bordercolor": "#0d2a40",
            "steps": [
                {"range": [0, 35],  "color": "#041208"},
                {"range": [35, 65], "color": "#0d0a00"},
                {"range": [65, 100],"color": "#1a0508"},
            ],
            "threshold": {"line": {"color": "rgba(255, 255, 255, 0.27)", "width": 2}, "thickness": 0.75, "value": 50}
        }
    ))
    fig.update_layout(
        height=200, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=0),
        font_color="#5a8aaa"
    )
    return fig


def make_timeline(frame_results) -> go.Figure:
    """Create timeline chart"""
    times  = [r.timestamp_sec for r in frame_results]
    fprobs = [r.fake_prob * 100 for r in frame_results]
    faces  = [r.face_detected  for r in frame_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=fprobs, fill="tozeroy",
        fillcolor="rgba(255,34,68,0.07)",
        line=dict(color="#ff2244", width=1.5),
        name="Fake %",
        hovertemplate="⏱ %{x:.1f}s  |  Fake: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255, 255, 255, 0.13)",
                  annotation_text="threshold", annotation_font_color="#3a6a8a",
                  annotation_font_size=9)
    
    for t, p, face in zip(times, fprobs, faces):
        if not face:
            fig.add_vline(x=t, line_width=0.5, line_color="rgba(255, 170, 0, 0.27)")

    fig.update_layout(
        title=dict(text="FRAME-BY-FRAME FAKE PROBABILITY", font=dict(family="Rajdhani", size=12, color="#3a6a8a")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a10",
        height=240, margin=dict(l=10, r=10, t=40, b=30),
        xaxis=dict(title="Time (s)", color="#3a6a8a", gridcolor="#0d1e2a", showgrid=True),
        yaxis=dict(title="Fake %", color="#3a6a8a", gridcolor="#0d1e2a", range=[0, 105]),
        font_color="#5a8aaa", showlegend=False,
    )
    return fig


def make_bar_chart(fake_p: float, real_p: float) -> go.Figure:
    """Create bar chart"""
    fig = go.Figure(go.Bar(
        x=["FAKE", "REAL"],
        y=[fake_p, real_p],
        marker_color=["#ff2244", "#00cc88"],
        text=[f"{fake_p:.1f}%", f"{real_p:.1f}%"],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a10",
        height=200, margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(range=[0, 115], gridcolor="#0d1e2a", color="#3a6a8a"),
        xaxis=dict(color="#3a6a8a"),
        font_color="#5a8aaa",
    )
    return fig


def show_result(result: str, fake_prob: float, confidence: str, extra: str = ""):
    """Display result card"""
    cls = "result-fake" if result == "Fake" else "result-real"
    icon = "⚠" if result == "Fake" else "✓"
    badge_cls = f"badge-{confidence}"
    st.markdown(f"""
    <div class="{cls}">
        <p class="result-label">{icon} {result.upper()}</p>
        <p class="result-meta">
            FAKE PROBABILITY: {fake_prob:.2f}% &nbsp;│&nbsp;
            CONFIDENCE: <span class="badge {badge_cls}">{confidence.upper()}</span>
            {f'&nbsp;│&nbsp; {extra}' if extra else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)


def generate_pdf_report(result, fake_prob, real_prob, confidence, report_type="image"):
    """Generate PDF report"""
    if not HAS_REPORTLAB:
        return None
    fname = "deepshield_report.pdf"
    try:
        c = rl_canvas.Canvas(fname)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(60, 780, "DeepShield — Detection Report")
        c.setFont("Helvetica", 12)
        c.drawString(60, 750, f"Type: {report_type.title()} Analysis")
        c.drawString(60, 730, f"Verdict: {result}")
        c.drawString(60, 710, f"Fake Probability: {fake_prob:.2f}%")
        c.drawString(60, 690, f"Real Probability: {real_prob:.2f}%")
        c.drawString(60, 670, f"Confidence: {confidence}")
        c.drawString(60, 640, "Generated by DeepShield v3.0 | BCA Final Year Project")
        c.save()
        return fname
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None


# ════════════════════════════════════════════════════════════
#  SIDEBAR (OPTIMIZED - DISABLED BY DEFAULT)
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;color:#00ffe0;
                font-size:1.1rem;letter-spacing:3px;
                text-transform:uppercase;margin-bottom:1rem;
                border-bottom:1px solid #0d2a40;padding-bottom:.5rem">
        ⚙ CONFIG
    </div>""", unsafe_allow_html=True)

    st.markdown("**Model**")
    model_choice = st.selectbox("Active Model",
        ["Calibrated Ensemble", "deepfake_final_model.pth", "deepfake_strong_model.pth", "deepfake_mvp_model.pth"],
        label_visibility="collapsed")

    st.markdown("**Video Settings**")
    n_frames  = st.slider("Frames to sample", 10, 60, 30)
    threshold = st.slider("Decision threshold", 0.3, 0.7, CALIBRATED_IMAGE_THRESHOLD, 0.01)
    use_tta   = st.checkbox("Test-Time Augmentation", value=False)
    use_face  = st.checkbox("Face detection", value=True)

    st.markdown("**Image Settings**")
    show_gradcam   = st.checkbox("GradCAM heatmap",   value=False)
    show_landmarks = st.checkbox("Face landmarks",     value=False)
    show_fft       = st.checkbox("FFT analysis",       value=False)
    show_forensics = st.checkbox("Forensic scores",    value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;
                font-size:0.7rem;color:#1a4a6a;line-height:1.8">
        CLASS MAPPING<br>
        ─────────────<br>
        class 0 = Fake<br>
        class 1 = Real<br>
        threshold = 0.595<br>
        arch = ResNet-18 Ensemble
    </div>""", unsafe_allow_html=True)

# Load model based on selection
model = load_image_ensemble() if model_choice == "Calibrated Ensemble" else load_image_model(model_choice)


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="ds-header">
    <p class="ds-title">Deep<span>Shield</span></p>
    <p class="ds-subtitle">
        AI-POWERED DEEPFAKE DETECTION ENGINE &nbsp;·&nbsp;
        FACEFORENSICS++ TRAINED &nbsp;·&nbsp; BCA FINAL YEAR PROJECT
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
if HAS_VIDEO_PREDICTOR:
    tab_img, tab_vid, tab_cam, tab_info = st.tabs([
        "🖼  IMAGE DETECTION",
        "🎬  VIDEO DETECTION",
        "📷  WEBCAM",
        "📊  SYSTEM INFO",
    ])
else:
    tab_img, tab_cam, tab_info = st.tabs([
        "🖼  IMAGE DETECTION",
        "📷  WEBCAM",
        "📊  SYSTEM INFO",
    ])


# ════════════════════════════════════════════════════════════
#  IMAGE TAB
# ════════════════════════════════════════════════════════════
with tab_img:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<div class="sec-head">Upload Image</div>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader(
            "JPG / PNG / WEBP", type=["jpg","jpeg","png","webp","bmp"],
            key="img_up", label_visibility="collapsed"
        )
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            analyze_img = st.button("⬡  RUN ANALYSIS", use_container_width=True, key="analyze_btn")

    with col_res:
        if uploaded_img and analyze_img:
            image = Image.open(uploaded_img).convert("RGB")
            with st.spinner("🔍 Analyzing image..."):
                face_img = detect_face_cv(image)
                result, conf, fake_p, real_p = predict_image(face_img, model)

            if result is not None:
                show_result(result, fake_p, "high" if conf > 80 else "medium" if conf > 60 else "low")
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-tile">
                        <p class="metric-val">{fake_p:.1f}%</p>
                        <p class="metric-lbl">Fake Prob</p>
                    </div>
                    <div class="metric-tile">
                        <p class="metric-val">{real_p:.1f}%</p>
                        <p class="metric-lbl">Real Prob</p>
                    </div>
                    <div class="metric-tile">
                        <p class="metric-val">{conf:.1f}%</p>
                        <p class="metric-lbl">Confidence</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(make_gauge(fake_p),
                        use_container_width=True, config={"displayModeBar": False})
                with c2:
                    st.plotly_chart(make_bar_chart(fake_p, real_p),
                        use_container_width=True, config={"displayModeBar": False})

                if show_gradcam:
                    st.markdown('<div class="sec-head">GradCAM Attention Map</div>', unsafe_allow_html=True)
                    cam = generate_gradcam(face_img, model)
                    st.image(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
                             caption="Regions influencing the decision", use_container_width=True)

                st.markdown('<div class="sec-head">Manipulation Heatmap</div>', unsafe_allow_html=True)
                hm = manipulation_heatmap(image)
                st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), use_container_width=True)

                if show_landmarks and HAS_MEDIAPIPE:
                    st.markdown('<div class="sec-head">Face Landmark Mesh</div>', unsafe_allow_html=True)
                    lm_img = face_landmarks_img(image)
                    st.image(lm_img, use_container_width=True)
                    st.caption(f"Eye status: {eye_blink_status(image)}")

                if show_fft:
                    st.markdown('<div class="sec-head">FFT Frequency Analysis</div>', unsafe_allow_html=True)
                    fft_img = fft_analysis(image)
                    st.image(cv2.cvtColor(fft_img, cv2.COLOR_BGR2RGB),
                             caption="GAN artifacts visible as bright frequency patterns",
                             use_container_width=True)

                if show_forensics:
                    st.markdown('<div class="sec-head">Forensic Scores</div>', unsafe_allow_html=True)
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        gan_score = gan_artifact_score(image)
                        st.metric("GAN Artifact Score", f"{gan_score:.3f}",
                                  help="Higher = more color channel inconsistency")
                    with fc2:
                        cons_score = face_consistency_score(image)
                        st.metric("Edge Consistency", f"{cons_score:.4f}",
                                  help="Higher = more edge artifacts")

                st.markdown("---")
                if st.button("📄 Download PDF Report"):
                    pdf = generate_pdf_report(result, fake_p, real_p,
                        "high" if conf > 80 else "medium", "image")
                    if pdf:
                        with open(pdf, "rb") as f:
                            st.download_button("⬇ Download", f,
                                file_name="deepshield_report.pdf", mime="application/pdf")
            else:
                st.error("❌ Model not loaded. Check console for errors.")


# ════════════════════════════════════════════════════════════
#  VIDEO TAB (Conditional)
# ════════════════════════════════════════════════════════════
if HAS_VIDEO_PREDICTOR:
    with tab_vid:
        col_up2, col_res2 = st.columns([1, 1], gap="large")

        with col_up2:
            st.markdown('<div class="sec-head">Upload Video</div>', unsafe_allow_html=True)
            uploaded_vid = st.file_uploader(
                "MP4 / AVI / MOV", type=["mp4","avi","mov","mkv","mpeg4"],
                key="vid_up", label_visibility="collapsed"
            )
            if uploaded_vid:
                st.video(uploaded_vid)
                analyze_vid = st.button("⬡  RUN VIDEO ANALYSIS", use_container_width=True, key="analyze_vid_btn")

        with col_res2:
            if uploaded_vid and analyze_vid:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_vid.getvalue())
                    tmp_path = tmp.name

                try:
                    st.markdown('<div class="sec-head">Analysis Progress</div>',
                                unsafe_allow_html=True)
                    prog = st.progress(0)
                    status = st.empty()

                    def cb(curr, total):
                        prog.progress(curr / total)
                        status.markdown(
                            f'<div style="font-family:\'Share Tech Mono\',monospace;'
                            f'font-size:0.75rem;color:#3a6a8a">'
                            f'SCANNING FRAME {curr}/{total}</div>',
                            unsafe_allow_html=True
                        )

                    t0 = time.time()
                    video_model_choice = "deepfake_final_model.pth" if model_choice == "Calibrated Ensemble" else model_choice
                    predictor = get_predictor(os.path.join(PROJECT_ROOT, video_model_choice))
                    predictor.threshold = threshold
                    predictor.n_frames  = n_frames
                    predictor.use_tta   = use_tta

                    vr = predictor.predict(tmp_path, progress_callback=cb)
                    elapsed = time.time() - t0

                    prog.empty()
                    status.empty()

                    show_result(
                        vr.result, vr.fake_prob * 100, vr.confidence,
                        f"STREAK: {'YES' if vr.streak_detected else 'NO'}"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    if vr.warning:
                        st.markdown(f'<div class="warn-box">⚠ {vr.warning}</div>',
                                    unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-tile">
                            <p class="metric-val">{vr.fake_prob*100:.1f}%</p>
                            <p class="metric-lbl">Fake Prob</p>
                        </div>
                        <div class="metric-tile">
                            <p class="metric-val">{vr.fake_frame_ratio*100:.0f}%</p>
                            <p class="metric-lbl">Fake Frames</p>
                        </div>
                        <div class="metric-tile">
                            <p class="metric-val">{vr.frames_analyzed}</p>
                            <p class="metric-lbl">Analyzed</p>
                        </div>
                        <div class="metric-tile">
                            <p class="metric-val">{vr.frames_with_faces}</p>
                            <p class="metric-lbl">Faces Found</p>
                        </div>
                        <div class="metric-tile">
                            <p class="metric-val">{elapsed:.0f}s</p>
                            <p class="metric-lbl">Time</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns([1.4, 0.6])
                    with c1:
                        if vr.frame_results:
                            st.plotly_chart(make_timeline(vr.frame_results),
                                use_container_width=True, config={"displayModeBar": False})
                    with c2:
                        st.plotly_chart(make_gauge(vr.fake_prob * 100),
                            use_container_width=True, config={"displayModeBar": False})

                    with st.expander("🔬 Temporal Analysis Details"):
                        t1, t2, t3 = st.columns(3)
                        t1.metric("Temporal Score", f"{vr.temporal_score:.3f}")
                        t2.metric("Streak Detected", "YES" if vr.streak_detected else "NO")
                        t3.metric("Fake Frame Ratio", f"{vr.fake_frame_ratio:.2%}")

                    with st.expander("📋 Per-Frame Results"):
                        rows = [{
                            "Frame": r.frame_idx,
                            "Time (s)": r.timestamp_sec,
                            "Fake %": f"{r.fake_prob*100:.1f}",
                            "Real %": f"{r.real_prob*100:.1f}",
                            "Face": "✓" if r.face_detected else "✗",
                            "Verdict": "FAKE" if r.fake_prob > threshold else "REAL",
                        } for r in vr.frame_results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

                    st.markdown("---")
                    if st.button("📄 Download Video Report"):
                        pdf = generate_pdf_report(vr.result, vr.fake_prob*100,
                            vr.real_prob*100, vr.confidence, "video")
                        if pdf:
                            with open(pdf, "rb") as f:
                                st.download_button("⬇ Download", f,
                                    file_name="deepshield_video_report.pdf",
                                    mime="application/pdf")

                except Exception as e:
                    st.error(f"❌ Video analysis error: {e}")
                    st.exception(e)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


# ════════════════════════════════════════════════════════════
#  WEBCAM TAB
# ════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown('<div class="sec-head">Live Webcam Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                color:#3a6a8a;line-height:1.8;margin-bottom:1rem">
        Real-time deepfake detection via webcam.<br>
        Capture a frame and analyze it instantly.
    </div>""", unsafe_allow_html=True)

    cam_col1, cam_col2 = st.columns([1, 1])
    with cam_col1:
        capture_btn = st.button("📷  CAPTURE FRAME", use_container_width=True, key="capture_btn")
        cam_img_placeholder = st.empty()

    with cam_col2:
        cam_result_placeholder = st.empty()

    if capture_btn:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                cam_img_placeholder.image(rgb, caption="Captured frame",
                                          use_container_width=True)
                with st.spinner("🔍 Analyzing..."):
                    face = detect_face_cv(pil)
                    result, conf, fake_p, real_p = predict_image(face, model)
                if result is not None:
                    with cam_result_placeholder.container():
                        show_result(result, fake_p,
                            "high" if conf > 80 else "medium" if conf > 60 else "low")
                        st.plotly_chart(make_gauge(fake_p),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.error("❌ Cannot access webcam. Check camera permissions.")

    st.markdown("---")
    st.info("For continuous real-time detection, run `webcam_detector.py` directly from terminal.")


# ════════════════════════════════════════════════════════════
#  SYSTEM INFO TAB
# ════════════════════════════════════════════════════════════
with tab_info:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        **Pipeline:**
        1. Face detection (Haar Cascade)
        2. Face crop + margin expansion
        3. ResNet-18 CNN classification (Ensemble)
        4. Test-Time Augmentation (3 views)
        5. Confidence-weighted aggregation
        6. Temporal streak analysis (video)
        7. Combined score → final decision

        **Class Mapping:**
        - `class 0 = Fake`
        - `class 1 = Real`

        **Decision Logic:**
        ```
        if fake_prob >= 0.595 → "Fake"
        if fake_prob < 0.595 → "Real"
        ```
        """)

    with c2:
        st.markdown('<div class="sec-head">System Status</div>', unsafe_allow_html=True)
        status_color = "✅" if model else "❌"
        st.info(f"{status_color} Image Detection: {'Ready' if model else 'Model Error'}")
        st.info(f"{'✅' if HAS_VIDEO_PREDICTOR else '⚠️'} Video Detection: {'Ready' if HAS_VIDEO_PREDICTOR else 'Module Not Found'}")
        st.info(f"{'✅' if HAS_MEDIAPIPE else '⚠️'} MediaPipe: {'Available' if HAS_MEDIAPIPE else 'Not Installed'}")
        st.info(f"{'✅' if HAS_REPORTLAB else '⚠️'} ReportLab: {'Available' if HAS_REPORTLAB else 'Not Installed'}")

    st.markdown('<div class="sec-head">Architecture Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=["ResNet-18\n(current)", "ResNet-50", "EfficientNet-B4"],
        y=[78, 88, 93],
        marker_color=["#3a6a8a", "#0066ff", "#00ffe0"],
        text=["~78%", "~88%", "~93%"],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=12, color="#c8d8e8"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060a10",
        height=250, margin=dict(l=10, r=10, t=10, b=40),
        yaxis=dict(range=[60, 100], gridcolor="#0d1e2a", color="#3a6a8a", title="AUC %"),
        xaxis=dict(color="#3a6a8a"),
        font_color="#5a8aaa",
        title=dict(text="MODEL ACCURACY COMPARISON", font=dict(family="Rajdhani", size=12, color="#3a6a8a")),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")
st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
            color:#1a4a6a;text-align:center;letter-spacing:2px">
    DEEPSHIELD v3.0 &nbsp;│&nbsp; BCA FINAL YEAR PROJECT &nbsp;│&nbsp;
    TRAINED ON FACEFORENSICS++ C23
</div>""", unsafe_allow_html=True)