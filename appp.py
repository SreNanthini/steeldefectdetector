"""
╔══════════════════════════════════════════════════════════════════╗
║     STEEL DEFECT DETECTION — STREAMLIT CLIENT DASHBOARD         ║
║     Intelligent Vision-Based Production Line Monitor            ║
║                                                                  ║
║  HOW TO RUN (2 terminals):                                      ║
║  ─────────────────────────────────────────────────────────────  ║
║  Terminal 1 → python server.py        (start Flask API)         ║
║  Terminal 2 → streamlit run app.py    (start Streamlit UI)      ║
║                                                                  ║
║  INSTALL (once):                                                 ║
║    pip install streamlit ultralytics opencv-python-headless     ║
║               torch torchvision pillow numpy pandas matplotlib  ║
║               flask flask-cors requests                          ║
║                                                                  ║
║  FILES NEEDED (same folder):                                     ║
║    app.py                  ← this file                           ║
║    server.py               ← Flask API backend                   ║
║    yolov8_best.pt          ← YOLOv8 weights (from Colab)        ║
║    rf_model.pkl            ← RandomForest model (from Colab)    ║
║    val_images.csv          ← dataset image paths (from Colab)   ║
║    resnet50_finetuned.pth  ← Grad-CAM model (optional)          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pickle
import requests
from PIL import Image
from datetime import datetime
from collections import deque

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Defect Detection — Production Line Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d3b5e 100%);
        padding: 20px 28px; border-radius: 12px; margin-bottom: 20px;
        border-left: 5px solid #00d4ff;
    }
    .main-header h1 { color: #00d4ff; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { color: #8eb8d4; margin: 4px 0 0 0; font-size: 0.9rem; }
    .metric-card {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 10px; padding: 18px 20px; text-align: center;
        transition: border-color 0.3s;
    }
    .metric-card:hover { border-color: #00d4ff; }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #00d4ff; }
    .metric-label { font-size: 0.78rem; color: #8eb8d4; margin-top: 4px; }
    .alert-critical {
        background: #3d1515; border: 2px solid #ff4444; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
    }
    .alert-warning {
        background: #3d2e0a; border: 2px solid #ffaa00; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
    }
    .alert-ok {
        background: #0d2e1a; border: 2px solid #00cc66; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
    }
    .api-online  { background:#0d2e1a; border:1px solid #00cc66; border-radius:8px;
                   padding:6px 14px; font-size:0.85rem; color:#00cc66; display:inline-block; }
    .api-offline { background:#3d1515; border:1px solid #ff4444; border-radius:8px;
                   padding:6px 14px; font-size:0.85rem; color:#ff4444; display:inline-block; }
    .badge-critical { background:#ff4444; color:#fff; padding:2px 10px;
                      border-radius:12px; font-size:0.8rem; font-weight:700; }
    .badge-major    { background:#ff8800; color:#fff; padding:2px 10px;
                      border-radius:12px; font-size:0.8rem; font-weight:700; }
    .badge-minor    { background:#ffcc00; color:#000; padding:2px 10px;
                      border-radius:12px; font-size:0.8rem; font-weight:700; }
    .badge-ok       { background:#00cc66; color:#fff; padding:2px 10px;
                      border-radius:12px; font-size:0.8rem; font-weight:700; }
    .section-header {
        color: #00d4ff; font-size: 1.1rem; font-weight: 700;
        border-bottom: 1px solid #2d3748; padding-bottom: 8px; margin-bottom: 16px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0066cc, #0099ff);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 8px 24px;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #0055aa, #0077dd); }
    div[data-testid="stSidebarContent"] { background: #1a1f2e; }
    .stSelectbox label, .stSlider label { color: #8eb8d4; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
COLORS  = {
    'crazing':         (231,  76,  60),
    'inclusion':       ( 46, 204, 113),
    'patches':         ( 52, 152, 219),
    'pitted_surface':  (243, 156,  18),
    'rolled-in_scale': (155,  89, 182),
    'scratches':       ( 26, 188, 156),
}
SEVERITY_MAP = {
    'crazing':          ('Minor',    'badge-minor'),
    'inclusion':        ('Major',    'badge-major'),
    'patches':          ('Minor',    'badge-minor'),
    'pitted_surface':   ('Critical', 'badge-critical'),
    'rolled-in_scale':  ('Major',    'badge-major'),
    'scratches':        ('Major',    'badge-major'),
}
ALERT_THRESHOLD = 5
API_BASE_URL    = "http://127.0.0.1:5000"

# ════════════════════════════════════════════════════════════════
# CLIENT-SERVER: FLASK API HELPERS
# ════════════════════════════════════════════════════════════════

def api_predict(image_bytes, conf=0.25, iou=0.45, filename="image.jpg"):
    """POST image to Flask /predict. Returns (detections, inference_ms, error)."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            files={"image": (filename, image_bytes, "image/jpeg")},
            data={"conf": conf, "iou": iou},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("detections", []), data.get("inference_ms", 0), None
        err = resp.json().get("error", f"HTTP {resp.status_code}")
        return [], 0, f"API error: {err}"
    except requests.exceptions.ConnectionError:
        return [], 0, "Flask server offline — start: python server.py"
    except requests.exceptions.Timeout:
        return [], 0, "Request timed out (>15s)"
    except Exception as e:
        return [], 0, f"Unexpected error: {str(e)}"


def api_health():
    """GET /health. Returns (is_online, info_dict)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except Exception:
        return False, None


def api_history(limit=20):
    """GET /history. Returns list of server-side records."""
    try:
        resp = requests.get(f"{API_BASE_URL}/history", params={"limit": limit}, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("records", [])
        return []
    except Exception:
        return []


def draw_boxes_from_api(img_bgr, detections):
    """Draw bounding boxes from API JSON onto img_bgr. Returns RGB image."""
    annotated = img_bgr.copy()
    for d in detections:
        cls_name        = d.get('class', 'unknown')
        conf            = float(d.get('confidence', 0))
        x1, y1, x2, y2 = [int(v) for v in d.get('box', [0, 0, 50, 50])]
        color           = COLORS.get(cls_name, (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'total_inspected': 0,
        'total_defective': 0,
        'defect_counts':   {c: 0 for c in CLASSES},
        'recent_defects':  deque(maxlen=60),
        'history':         [],
        'running':         False,
        'start_time':      None,
        'frame_times':     deque(maxlen=30),
        'alerts':          [],
        'availability':    0.95,
        'performance':     0.90,
        'api_online':      False,
        'total_api_calls': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════════
# LOCAL MODEL LOADING (for simulation + Grad-CAM)
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading local YOLOv8 (simulation mode)...")
def load_yolo(path='yolov8_best.pt'):
    try:
        from ultralytics import YOLO
        return YOLO(path), None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="Loading ResNet50 + RandomForest...")
def load_rf(path='rf_model.pkl'):
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d['rf'], d['scaler'], None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(show_spinner="Loading dataset CSV...")
def load_dataset(csv_path='val_images.csv'):
    try:
        df = pd.read_csv(csv_path)
        df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

def run_yolo_local(model, img_bgr, conf_thresh=0.25, iou_thresh=0.45):
    """Local YOLO inference — used for simulation mode only."""
    results    = model.predict(img_bgr, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
    detections = []
    annotated  = img_bgr.copy()
    for r in results:
        for box in (r.boxes or []):
            cls_id   = int(box.cls)
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else 'unknown'
            conf     = float(box.conf)
            xyxy     = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({'class': cls_name, 'confidence': conf, 'box': xyxy.tolist()})
            color    = COLORS.get(cls_name, (255, 255, 0))
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label    = f'{cls_name} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1] - th - 6), (xyxy[0] + tw + 4, xyxy[1]), color, -1)
            cv2.putText(annotated, label, (xyxy[0] + 2, xyxy[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), detections

# ── OEE / alert helpers ───────────────────────────────────────────
def compute_oee():
    s            = st.session_state
    availability = s['availability']
    performance  = s['performance']
    total        = s['total_inspected']
    defective    = s['total_defective']
    quality      = (total - defective) / total if total > 0 else 1.0
    return availability, performance, quality, availability * performance * quality

def defects_per_minute():
    now    = time.time()
    recent = [t for t in st.session_state['recent_defects'] if now - t <= 60]
    st.session_state['recent_defects'] = deque(recent, maxlen=60)
    return len(recent)

def check_alerts(dpm, defect_list):
    alerts  = st.session_state['alerts']
    now_str = datetime.now().strftime('%H:%M:%S')
    if dpm >= ALERT_THRESHOLD:
        msg = f"[{now_str}] 🚨 HIGH DEFECT RATE: {dpm} defects/min (threshold: {ALERT_THRESHOLD})"
        if not alerts or alerts[-1] != msg:
            alerts.append(msg)
    for d in defect_list:
        if d['class'] == 'pitted_surface' and d['confidence'] > 0.6:
            alerts.append(f"[{now_str}] ⚠️ CRITICAL defect: {d['class']} (conf: {d['confidence']:.2f})")
    st.session_state['alerts'] = alerts[-10:]

# ── Grad-CAM ──────────────────────────────────────────────────────
def generate_gradcam(img_path, model_path='resnet50_finetuned.pth'):
    import torch, torch.nn as nn
    import torchvision.models as tvmodels
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    base    = tvmodels.resnet50(weights=None)
    base.fc = nn.Linear(2048, 6)
    if os.path.exists(model_path):
        try:
            base.load_state_dict(torch.load(model_path, map_location='cpu'))
        except Exception:
            pass
    base.eval()

    activations, gradients = {}, {}
    base.layer4[-1].register_forward_hook(lambda m, i, o: activations.update({'feat': o.detach()}))
    base.layer4[-1].register_full_backward_hook(lambda m, gi, go: gradients.update({'grad': go[0].detach()}))

    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        img_gray = np.zeros((200, 200), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    tensor  = transform(img_rgb).unsqueeze(0)

    out     = base(tensor)
    cls_idx = out.argmax(1).item()
    base.zero_grad()
    out[0, cls_idx].backward()

    w   = gradients['grad'].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((w * activations['feat']).sum(dim=1)).squeeze().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    img_show = cv2.resize(img_rgb, (224, 224))
    heatmap  = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(img_show, 0.55, heatmap, 0.45, 0)
    return overlay, CLASSES[cls_idx], float(torch.softmax(out, dim=1)[0, cls_idx])

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-header">🌐 Flask API Server</p>', unsafe_allow_html=True)
    api_url_input = st.text_input("API Base URL", value=API_BASE_URL)
    API_BASE_URL  = api_url_input.rstrip("/")

    if st.button("🔍 Check Server Status", use_container_width=True):
        is_online, info = api_health()
        st.session_state['api_online'] = is_online
        if is_online:
            st.success(f"✅ Online | Model: {'✅' if info.get('model_loaded') else '❌'} | "
                       f"Requests: {info.get('total_requests', 0)}")
        else:
            st.error("❌ Offline — run: python server.py")

    badge = 'api-online' if st.session_state['api_online'] else 'api-offline'
    text  = '🟢 Flask API: ONLINE' if st.session_state['api_online'] else '🔴 Flask API: OFFLINE'
    st.markdown(f'<div class="{badge}">{text}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">📁 Local Model Paths</p>', unsafe_allow_html=True)
    yolo_path   = st.text_input("YOLOv8 model path",       value="yolov8_best.pt")
    rf_path     = st.text_input("RF model path",            value="rf_model.pkl")
    csv_path    = st.text_input("Dataset CSV path",         value="val_images.csv")
    resnet_path = st.text_input("ResNet50 path (Grad-CAM)", value="resnet50_finetuned.pth")

    st.markdown("---")
    st.markdown('<p class="section-header">🎛️ Detection Settings</p>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    iou_thresh  = st.slider("IoU Threshold",        0.1, 0.9, 0.45, 0.05)
    delay_ms    = st.slider("Frame Delay (ms)",      50, 2000, 400, 50)
    max_frames  = st.slider("Max frames per run",    10, 200,   50)

    st.markdown("---")
    st.markdown('<p class="section-header">📊 OEE Simulation</p>', unsafe_allow_html=True)
    st.session_state['availability'] = st.slider("Availability", 0.5, 1.0, 0.95, 0.01)
    st.session_state['performance']  = st.slider("Performance",  0.5, 1.0, 0.90, 0.01)

    st.markdown("---")
    st.caption(f"API calls made this session: **{st.session_state['total_api_calls']}**")

    if st.button("🔄 Reset All Metrics"):
        for k in ['total_inspected', 'total_defective', 'defect_counts',
                  'recent_defects', 'history', 'alerts', 'frame_times', 'total_api_calls']:
            if k == 'defect_counts':
                st.session_state[k] = {c: 0 for c in CLASSES}
            elif k in ['recent_defects', 'frame_times']:
                st.session_state[k] = deque(maxlen=60)
            elif k in ['history', 'alerts']:
                st.session_state[k] = []
            else:
                st.session_state[k] = 0
        st.session_state['running'] = False
        st.rerun()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏭 Steel Surface Defect Detection System</h1>
    <p>Intelligent Vision-Based Production Line Monitor | NEU-DET Dataset | YOLOv8 + ResNet50 | Client-Server Architecture</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Live Detection", "📊 Analytics", "🔬 Grad-CAM (XAI)",
    "📋 Inspection Log", "🌐 API Monitor", "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DETECTION
# ════════════════════════════════════════════════════════════════
with tab1:
    col_feed, col_panel = st.columns([3, 2], gap="medium")

    with col_feed:
        st.markdown('<p class="section-header">📷 Inspection Feed</p>', unsafe_allow_html=True)
        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            start_btn = st.button("▶️ Start Inspection", use_container_width=True)
        with ctrl2:
            stop_btn  = st.button("⏹ Stop",             use_container_width=True)
        with ctrl3:
            single_btn = st.button("📸 Single Frame",   use_container_width=True)

        st.markdown("---")
        st.markdown("**📤 Upload image — processed by Flask API:**")
        uploaded = st.file_uploader(
            "Upload steel surface image (sent to Flask server):",
            type=['jpg', 'png', 'jpeg']
        )

    with col_panel:
        st.markdown('<p class="section-header">📈 Live Metrics</p>', unsafe_allow_html=True)
        met1, met2 = st.columns(2)
        met3, met4 = st.columns(2)
        metric_placeholders = {
            'inspected':   met1.empty(),
            'defective':   met2.empty(),
            'defect_rate': met3.empty(),
            'fps':         met4.empty(),
        }
        oee_placeholder   = st.empty()
        alert_placeholder = st.empty()
        detect_placeholder = st.empty()

    # Load local models
    yolo_model, yolo_err       = load_yolo(yolo_path)
    rf_model, rf_scaler, rf_err = load_rf(rf_path)
    df_dataset, ds_err         = load_dataset(csv_path)

    def update_metrics():
        s         = st.session_state
        total     = s['total_inspected']
        defective = s['total_defective']
        rate      = defective / total * 100 if total > 0 else 0
        fps       = 1.0 / np.mean(list(s['frame_times'])) if s['frame_times'] else 0
        avail, perf, qual, oee = compute_oee()
        dpm = defects_per_minute()

        with metric_placeholders['inspected'].container():
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div>'
                        f'<div class="metric-label">Total Inspected</div></div>', unsafe_allow_html=True)

        with metric_placeholders['defective'].container():
            col = '#ff4444' if rate > 10 else '#ffaa00' if rate > 5 else '#00cc66'
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{col}">'
                        f'{defective}</div><div class="metric-label">Defective Units</div></div>',
                        unsafe_allow_html=True)

        with metric_placeholders['defect_rate'].container():
            st.markdown(f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div>'
                        f'<div class="metric-label">Defect Rate</div></div>', unsafe_allow_html=True)

        with metric_placeholders['fps'].container():
            st.markdown(f'<div class="metric-card"><div class="metric-value">{fps:.1f}</div>'
                        f'<div class="metric-label">FPS (simulated)</div></div>', unsafe_allow_html=True)

        with oee_placeholder.container():
            st.markdown('<p class="section-header">⚙️ OEE Dashboard</p>', unsafe_allow_html=True)
            oee_pct = oee * 100
            col     = '#00cc66' if oee_pct >= 85 else '#ffaa00' if oee_pct >= 65 else '#ff4444'
            st.markdown(f'<div class="metric-card" style="margin-bottom:10px">'
                        f'<div class="metric-value" style="color:{col}">{oee_pct:.1f}%</div>'
                        f'<div class="metric-label">Overall Equipment Effectiveness</div></div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Availability", f"{avail*100:.1f}%")
            c2.metric("Performance",  f"{perf*100:.1f}%")
            c3.metric("Quality",      f"{qual*100:.1f}%")

        check_alerts(dpm, [])
        alerts = st.session_state['alerts']
        if alerts:
            with alert_placeholder.container():
                st.markdown('<p class="section-header">🚨 Alerts</p>', unsafe_allow_html=True)
                for a in reversed(alerts[-3:]):
                    cls = 'alert-critical' if '🚨' in a else 'alert-warning' if '⚠️' in a else 'alert-ok'
                    st.markdown(f'<div class="{cls}">{a}</div>', unsafe_allow_html=True)
        else:
            with alert_placeholder.container():
                st.markdown('<div class="alert-ok">✅ System Normal — No active alerts</div>',
                            unsafe_allow_html=True)

    def record_detection(img_name, detections, source='local'):
        s = st.session_state
        s['total_inspected'] += 1
        if detections:
            s['total_defective'] += 1
            for d in detections:
                s['defect_counts'][d['class']] = s['defect_counts'].get(d['class'], 0) + 1
                s['recent_defects'].append(time.time())
            check_alerts(defects_per_minute(), detections)
        s['history'].append({
            'time':        datetime.now().strftime('%H:%M:%S'),
            'image':       img_name,
            'defects':     len(detections),
            'classes':     [d['class'] for d in detections],
            'confidences': [f"{d['confidence']:.2f}" for d in detections],
            'source':      source,
        })

    def process_image_local(img_path):
        """For simulation — local YOLO inference."""
        t0      = time.time()
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None, []
        annotated_rgb, detections = run_yolo_local(yolo_model, img_bgr, conf_thresh, iou_thresh)
        st.session_state['frame_times'].append(time.time() - t0)
        record_detection(os.path.basename(img_path), detections, source='local')
        return annotated_rgb, detections

    # ── UPLOAD → FLASK API ─────────────────────────────────────
    if uploaded:
        img_pil = Image.open(uploaded).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("📡 Sending image to Flask API..."):
            t_start = time.time()
            detections, infer_ms, api_err = api_predict(
                uploaded.getvalue(), conf=conf_thresh, iou=iou_thresh, filename=uploaded.name)
            round_trip_ms = round((time.time() - t_start) * 1000, 1)

        if api_err:
            st.warning(f"⚠️ API unavailable: {api_err}")
            if yolo_model:
                st.info("↩️ Falling back to local YOLOv8...")
                annotated_rgb, detections = run_yolo_local(yolo_model, img_bgr, conf_thresh, iou_thresh)
                caption = f"{uploaded.name} | Local YOLOv8 (API offline)"
            else:
                frame_placeholder.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                        caption="API offline — no local model loaded",
                                        use_container_width=True)
                detect_placeholder.error(api_err)
                st.stop()
        else:
            st.session_state['api_online']      = True
            st.session_state['total_api_calls'] += 1
            annotated_rgb = draw_boxes_from_api(img_bgr, detections)
            caption       = (f"{uploaded.name} | ✅ Flask API | "
                             f"server: {infer_ms}ms | round-trip: {round_trip_ms}ms")

        record_detection(uploaded.name, detections, source='api')
        frame_placeholder.image(annotated_rgb, caption=caption, use_container_width=True)
        update_metrics()

        with detect_placeholder.container():
            st.markdown('<p class="section-header">🔍 Detections (Flask API)</p>',
                        unsafe_allow_html=True)
            if detections:
                for d in detections:
                    sev, badge = SEVERITY_MAP.get(d['class'], ('Unknown', 'badge-minor'))
                    st.markdown(f'<span class="{badge}">{sev}</span>'
                                f'&nbsp;&nbsp;<b>{d["class"]}</b> — conf: {d["confidence"]:.2f}',
                                unsafe_allow_html=True)
            else:
                st.success("✅ No defects detected")

    if yolo_err and not uploaded:
        st.warning(f"Local YOLOv8 not loaded: {yolo_err}")
    if ds_err:
        st.warning(f"Dataset CSV error: {ds_err} — simulation disabled")

    # ── Single Frame (local YOLO) ─────────────────────────────
    if single_btn and yolo_model and df_dataset is not None and len(df_dataset) > 0:
        row           = df_dataset.sample(1).iloc[0]
        annotated_rgb, detections = process_image_local(row['image_path'])
        if annotated_rgb is not None:
            frame_placeholder.image(
                annotated_rgb, use_container_width=True,
                caption=f"{os.path.basename(row['image_path'])} | GT: {row.get('label','?')} | Local")
            update_metrics()
            with detect_placeholder.container():
                st.markdown('<p class="section-header">🔍 Detections (Local YOLOv8)</p>',
                            unsafe_allow_html=True)
                if detections:
                    for d in detections:
                        sev, badge = SEVERITY_MAP.get(d['class'], ('Unknown', 'badge-minor'))
                        st.markdown(f'<span class="{badge}">{sev}</span>'
                                    f'&nbsp;&nbsp;<b>{d["class"]}</b> — conf: {d["confidence"]:.2f}',
                                    unsafe_allow_html=True)
                else:
                    st.success("✅ No defects detected")

    # ── Continuous simulation ─────────────────────────────────
    if start_btn:
        st.session_state['running']    = True
        st.session_state['start_time'] = time.time()
    if stop_btn:
        st.session_state['running'] = False

    if st.session_state['running'] and yolo_model and df_dataset is not None:
        frames_run   = 0
        progress_bar = st.progress(0)
        while st.session_state['running'] and frames_run < max_frames:
            row           = df_dataset.sample(1).iloc[0]
            annotated_rgb, detections = process_image_local(row['image_path'])
            if annotated_rgb is not None:
                ts = datetime.now().strftime('%H:%M:%S')
                frame_placeholder.image(
                    annotated_rgb, use_container_width=True,
                    caption=f"[{ts}] Frame {frames_run+1}/{max_frames} | "
                            f"GT: {row.get('label','?')} | Detected: {len(detections)}")
                with detect_placeholder.container():
                    st.markdown('<p class="section-header">🔍 Latest Detections</p>',
                                unsafe_allow_html=True)
                    if detections:
                        for d in detections[:4]:
                            sev, badge = SEVERITY_MAP.get(d['class'], ('Unknown', 'badge-minor'))
                            st.markdown(f'<span class="{badge}">{sev}</span>'
                                        f'&nbsp;&nbsp;<b>{d["class"]}</b> — conf: {d["confidence"]:.2f}',
                                        unsafe_allow_html=True)
                    else:
                        st.success("✅ No defects in this frame")
            update_metrics()
            frames_run += 1
            progress_bar.progress(frames_run / max_frames)
            time.sleep(delay_ms / 1000.0)

        st.session_state['running'] = False
        status_placeholder.success(f"✅ Inspection complete — {frames_run} frames processed")
        progress_bar.empty()

# ════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📊 Production Analytics Dashboard</p>',
                unsafe_allow_html=True)

    s                        = st.session_state
    total                    = s['total_inspected']
    defective                = s['total_defective']
    counts                   = s['defect_counts']
    avail, perf, qual, oee   = compute_oee()

    if total == 0:
        st.info("▶️ Run the inspection in **Live Detection** tab first to populate analytics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected", total)
        c2.metric("Defective Units", defective)
        c3.metric("Quality Rate",    f"{qual*100:.1f}%")
        c4.metric("OEE",             f"{oee*100:.1f}%")

        st.markdown("---")
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("**Defect Class Distribution**")
            active = {k: v for k, v in counts.items() if v > 0}
            if active:
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1f2e')
                ax.set_facecolor('#1a1f2e')
                bcolors = ['#E74C3C','#2ECC71','#3498DB','#F39C12','#9B59B6','#1ABC9C']
                bars    = ax.bar(list(active.keys()), list(active.values()),
                                 color=[bcolors[CLASSES.index(k)] for k in active], edgecolor='white')
                for b, v in zip(bars, active.values()):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                            str(v), ha='center', color='white', fontweight='bold', fontsize=10)
                ax.set_ylabel('Count', color='#8eb8d4')
                ax.tick_params(axis='x', rotation=25, colors='#8eb8d4')
                ax.tick_params(axis='y', colors='#8eb8d4')
                for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')
                ax.grid(axis='y', alpha=0.2, color='white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No defects detected yet")

        with col_b:
            st.markdown("**Pass / Fail Ratio**")
            passed = total - defective
            if total > 0:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1f2e')
                ax.set_facecolor('#1a1f2e')
                ax.pie([passed, defective],
                       labels=[f'Pass ({passed})', f'Fail ({defective})'],
                       colors=['#00cc66', '#ff4444'],
                       autopct='%1.1f%%', startangle=90,
                       wedgeprops=dict(width=0.5, edgecolor='#1a1f2e', linewidth=2),
                       textprops={'color': 'white'})
                ax.set_title('Inspection Result', color='#8eb8d4')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        st.markdown("**OEE Components**")
        g1, g2, g3, g4 = st.columns(4)
        for col, label, val, tg, tw in [
            (g1, "Availability", avail, 0.90, 0.75),
            (g2, "Performance",  perf,  0.90, 0.75),
            (g3, "Quality",      qual,  0.90, 0.75),
            (g4, "OEE",          oee,   0.85, 0.65),
        ]:
            color = '#00cc66' if val >= tg else '#ffaa00' if val >= tw else '#ff4444'
            col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                         f'{val*100:.1f}%</div><div class="metric-label">{label}</div></div>',
                         unsafe_allow_html=True)

        if any(v > 0 for v in counts.values()):
            st.markdown("---")
            st.markdown("**Defect Severity Breakdown**")
            sev_rows = []
            for cls, cnt in counts.items():
                if cnt > 0:
                    sev, _ = SEVERITY_MAP.get(cls, ('Minor', ''))
                    sev_rows.append({'Defect Class': cls, 'Count': cnt, 'Severity': sev,
                                     'Action': {'Critical':'🛑 Stop line','Major':'⚠️ Rework',
                                                'Minor':'📋 Monitor'}.get(sev,'—')})
            if sev_rows:
                st.dataframe(pd.DataFrame(sev_rows).sort_values('Count', ascending=False),
                             use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — GRAD-CAM
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">🔬 Grad-CAM — Explainable AI</p>',
                unsafe_allow_html=True)
    st.markdown("**Grad-CAM** highlights which image regions most influenced the model's prediction.")

    df_gc, gc_err = load_dataset(csv_path)
    uploaded_gc   = st.file_uploader("Upload image for Grad-CAM:", type=['jpg','png','jpeg'],
                                      key='gradcam_upload')
    col_gc1, col_gc2 = st.columns([1, 1], gap="medium")

    def show_gradcam(img_path_or_file):
        with st.spinner("Generating Grad-CAM heatmap..."):
            if hasattr(img_path_or_file, 'read'):
                img_pil = Image.open(img_path_or_file).convert('RGB')
                tmp     = '/tmp/gradcam_input.jpg'
                img_pil.save(tmp)
                use_path = tmp
            else:
                use_path = img_path_or_file
            overlay, pred_cls, conf = generate_gradcam(use_path, resnet_path)
            sev, badge = SEVERITY_MAP.get(pred_cls, ('Unknown', 'badge-minor'))
            with col_gc1:
                orig = cv2.imread(use_path, cv2.IMREAD_GRAYSCALE)
                if orig is not None:
                    st.image(cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB),
                             caption="Original Image", use_container_width=True)
            with col_gc2:
                st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
                st.markdown(f"**Prediction:** `{pred_cls}` | **Confidence:** `{conf*100:.1f}%` | "
                            f"**Severity:** <span class='{badge}'>{sev}</span>",
                            unsafe_allow_html=True)
                st.info("🔴 Red = high importance  |  🔵 Blue = low importance")

    if uploaded_gc:
        show_gradcam(uploaded_gc)
    elif st.button("🎲 Random sample from dataset") and df_gc is not None and len(df_gc) > 0:
        cls_filter = st.selectbox("Filter by class", ['All'] + CLASSES, key='gc_cls')
        sub        = df_gc[df_gc['label'] == cls_filter] if cls_filter != 'All' else df_gc
        if len(sub) > 0:
            show_gradcam(sub.sample(1).iloc[0]['image_path'])
    elif df_gc is None:
        st.info("Upload an image above, or fix the dataset CSV path in the sidebar.")

    if st.session_state['history']:
        st.markdown("---")
        st.markdown("**Recently inspected images:**")
        for h in reversed(st.session_state['history'][-5:]):
            if h['defects'] > 0:
                st.write(f"🕐 {h['time']} — `{h['image']}` — {h['classes']} — via {h.get('source','?')}")

# ════════════════════════════════════════════════════════════════
# TAB 4 — INSPECTION LOG
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">📋 Inspection Log</p>', unsafe_allow_html=True)
    history = st.session_state['history']
    if not history:
        st.info("No inspection records yet. Run the Live Detection to populate the log.")
    else:
        df_log    = pd.DataFrame(history)
        csv_bytes = df_log.to_csv(index=False).encode()
        st.download_button("⬇️ Download Log (CSV)", csv_bytes,
                           file_name=f"inspection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime='text/csv')
        st.markdown(f"**{len(history)} records** | Last: {history[-1]['time']}")

        df_display               = pd.DataFrame(history[-50:]).iloc[::-1]
        df_display['status']     = df_display['defects'].apply(lambda x: '✅ PASS' if x == 0 else '❌ FAIL')
        df_display['classes_str']= df_display['classes'].apply(lambda x: ', '.join(x) if x else '—')
        cols = ['time','image','status','defects','classes_str','confidences']
        if 'source' in df_display.columns:
            cols.append('source')
        st.dataframe(df_display[cols], use_container_width=True, hide_index=True,
                     column_config={
                         'time':        st.column_config.TextColumn('Time'),
                         'image':       st.column_config.TextColumn('Image'),
                         'status':      st.column_config.TextColumn('Status'),
                         'defects':     st.column_config.NumberColumn('# Defects'),
                         'classes_str': st.column_config.TextColumn('Defect Types'),
                         'confidences': st.column_config.TextColumn('Confidence'),
                         'source':      st.column_config.TextColumn('Source'),
                     })

# ════════════════════════════════════════════════════════════════
# TAB 5 — API MONITOR (NEW)
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">🌐 Flask API Monitor</p>', unsafe_allow_html=True)
    st.markdown("Live status of the **Flask backend server** that handles YOLOv8 inference.")

    # Status cards
    is_online, health_info = api_health()
    st.session_state['api_online'] = is_online
    m1, m2, m3 = st.columns(3)

    with m1:
        col    = '#00cc66' if is_online else '#ff4444'
        status = 'ONLINE' if is_online else 'OFFLINE'
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{col}">'
                    f'{status}</div><div class="metric-label">Flask Server</div></div>',
                    unsafe_allow_html=True)
    with m2:
        model_ok = health_info.get('model_loaded', False) if health_info else False
        col      = '#00cc66' if model_ok else '#ff4444'
        txt      = 'LOADED' if model_ok else 'NOT LOADED'
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{col};font-size:1.6rem">'
                    f'{txt}</div><div class="metric-label">YOLOv8 Model</div></div>',
                    unsafe_allow_html=True)
    with m3:
        req_count = health_info.get('total_requests', 0) if health_info else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{req_count}</div>'
                    f'<div class="metric-label">API Requests Served</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Endpoint table
    st.markdown("### 📋 API Endpoints")
    st.dataframe(pd.DataFrame({
        'Method':    ['POST', 'GET', 'GET', 'GET', 'DELETE'],
        'Endpoint':  ['/predict', '/health', '/classes', '/history', '/history'],
        'Description': [
            'Upload image → returns detections JSON',
            'Server + model status check',
            'List all 6 defect classes',
            'Last N prediction records',
            'Clear prediction history',
        ],
        'Called by Streamlit': ['✅ Upload button', '✅ Status check', '—', '✅ Below table', '—'],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🕐 Server-Side Prediction History")
    col_r1, col_r2 = st.columns([1, 4])
    with col_r1:
        hist_lim = st.number_input("Records", min_value=5, max_value=100, value=20)
    with col_r2:
        if st.button("🔄 Refresh"):
            st.rerun()

    srv_hist = api_history(limit=int(hist_lim))
    if srv_hist:
        st.dataframe(pd.DataFrame(srv_hist), use_container_width=True, hide_index=True)
    elif is_online:
        st.info("No server predictions yet. Upload an image to create one.")
    else:
        st.error("❌ Flask server is offline. Start with: python server.py")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start Commands")
    st.code("""# Install all dependencies
pip install streamlit ultralytics opencv-python torch torchvision
pip install flask flask-cors requests pillow numpy pandas matplotlib

# Terminal 1 — start Flask API server (port 5000)
python server.py

# Terminal 2 — start Streamlit client (port 8501)
streamlit run app.py

# Test API manually
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict -F "image=@test.jpg"
curl http://127.0.0.1:5000/history?limit=5
""", language="bash")

    st.markdown("### 🏗️ Client-Server Architecture")
    st.code("""
  ┌──────────────────────────────────┐         HTTP POST /predict
  │   STREAMLIT CLIENT  (port 8501) │ ──────────────────────────►  ┌─────────────────────────┐
  │   app.py                         │    { image: bytes,            │  FLASK SERVER (port 5000)│
  │                                  │      conf: 0.25, iou: 0.45 }  │  server.py               │
  │   • Tab 1: Upload → api_predict()│                              │                         │
  │   • Tab 5: API Monitor           │ ◄──────────────────────────  │  POST /predict           │
  │   • All tabs: display results    │    JSON: [{class, conf, box}] │    → YOLOv8 inference    │
  │   • Simulation: local YOLO       │                              │    → returns detections  │
  └──────────────────────────────────┘         GET /health          │  GET  /health            │
                                               GET /history          │  GET  /history           │
                                                                     └─────────────────────────┘
""", language="text")

# ════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
## About This Project

**Title:** Intelligent Vision-Based System for Production Line Performance Monitoring using Machine Learning
**Author:** V.S. SRE NANTHINI — 810022104026
**Guide:** Dr. K. LATHA, Dept. of Computer Science & Engineering

---

### System Architecture (Client-Server)

| Component | Technology | Role |
|-----------|-----------|------|
| **Client UI** | **Streamlit** (port 8501) | Dashboard, upload, OEE display |
| **API Server** | **Flask REST API** (port 5000) | Receives images, runs YOLOv8, returns JSON |
| **Object Detection** | **YOLOv8s** | Detect + localize defects with bounding boxes |
| **Feature Extraction** | **ResNet50** (frozen) | 2048-dim feature vectors for classification |
| **Classification** | **Random Forest** | 97.1% accuracy defect classification |
| **Explainability** | **Grad-CAM** | Heatmaps showing which regions the model attended to |

---

### Dataset — NEU Surface Defect Dataset

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Crazing | 205 | 60 | 36 |
| Inclusion | 214 | 60 | 38 |
| Patches | 247 | 62 | 44 |
| Pitted Surface | 306 | 58 | 54 |
| Rolled-in Scale | 204 | 60 | 36 |
| Scratches | 196 | 60 | 35 |

---

### Results Summary

| Model | Task | Key Metric | Score |
|-------|------|-----------|-------|
| YOLOv8s | Detection + Localization | mAP@50 | 72.1% |
| ResNet50 + RF | Classification | Accuracy | **97.1%** |

---

### Defect Severity

| Severity | Defect Types | Action |
|----------|-------------|--------|
| 🔴 Critical | Pitted Surface | Stop line immediately |
| 🟠 Major | Inclusion, Rolled-in Scale, Scratches | Flag for rework |
| 🟡 Minor | Crazing, Patches | Log and monitor |

---

### OEE Formula

```
OEE = Availability × Performance × Quality
Quality = (Total Inspected − Defective) / Total Inspected
```

---

### Key Features
- ✅ **Client-Server**: Streamlit ↔ Flask REST API (separation of concerns)
- ✅ Real-time detection via Flask API + bounding box rendering from JSON
- ✅ Production line simulation with local YOLOv8
- ✅ ResNet50 + Random Forest classification (97.1%)
- ✅ Grad-CAM Explainable AI heatmaps
- ✅ OEE dashboard (Availability × Performance × Quality)
- ✅ Severity alerts (Critical / Major / Minor)
- ✅ API Monitor tab — live server status and prediction history
- ✅ Graceful fallback to local model if API is offline
- ✅ Downloadable inspection logs (CSV)
""")
