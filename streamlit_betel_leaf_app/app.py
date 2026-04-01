# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os, io, csv, time, pandas as pd
import traceback
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------
# Page config & style
# -----------------------
st.set_page_config(page_title="Betel Leaf Disease Detector", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root {
  --primary:       #1a5c30;
  --primary-light: #2e8b57;
  --primary-dark:  #0f3d1e;
  --accent:        #a8d5b5;
  --accent2:       #f0c060;
  --bg:            #f4f8f5;
  --surface:       #ffffff;
  --surface2:      #eef6f1;
  --text:          #0e2d17;
  --text-muted:    #5a7a62;
  --border:        #d0e6d8;
  --danger:        #c0392b;
  --warn-bg:       #fef9ec;
  --warn-border:   #e8c84a;
  --radius:        14px;
  --radius-sm:     8px;
  --shadow:        0 4px 24px rgba(15,61,30,0.08);
  --shadow-hover:  0 8px 32px rgba(15,61,30,0.15);
  --transition:    all 0.25s cubic-bezier(0.4,0,0.2,1);
}

/* ── Base ── */
html, body, .stApp {
  background-color: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Top brand bar ── */
.brand-bar {
  background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 60%, var(--primary-light) 100%);
  padding: 18px 32px;
  border-radius: 0 0 var(--radius) var(--radius);
  margin-bottom: 28px;
  display: flex;
  align-items: center;
  gap: 14px;
  box-shadow: var(--shadow);
}
.brand-icon {
  font-size: 36px;
  line-height: 1;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
}
.brand-title {
  font-family: 'DM Serif Display', serif;
  font-size: 26px;
  color: #ffffff;
  letter-spacing: -0.3px;
  line-height: 1.1;
}
.brand-sub {
  font-size: 13px;
  color: rgba(255,255,255,0.72);
  font-weight: 300;
  margin-top: 2px;
  letter-spacing: 0.4px;
}

/* ── Tab panel acts as the card ── */
[data-testid="stTabsContent"] > div[role="tabpanel"] {
  background: var(--surface) !important;
  border-radius: 0 var(--radius) var(--radius) var(--radius) !important;
  padding: 28px 32px !important;
  box-shadow: var(--shadow) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  margin-bottom: 20px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  color: var(--text-muted) !important;
  border: none !important;
  padding: 10px 18px !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  transition: var(--transition) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--primary) !important;
  background: var(--surface) !important;
  border-bottom: 2px solid var(--primary) !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] button:hover {
  color: var(--primary-light) !important;
  background: var(--surface2) !important;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 28px 32px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  margin-bottom: 20px;
  transition: var(--transition);
}
.card:hover {
  box-shadow: var(--shadow-hover);
}
.card-sm {
  background: var(--surface2);
  border-radius: var(--radius-sm);
  padding: 16px 20px;
  border: 1px solid var(--border);
  margin-bottom: 12px;
}

/* ── Section headers ── */
.page-title {
  font-family: 'DM Serif Display', serif;
  font-size: 28px;
  color: var(--primary-dark);
  letter-spacing: -0.5px;
  margin-bottom: 4px;
  line-height: 1.15;
}
.page-sub {
  font-size: 14px;
  color: var(--text-muted);
  font-weight: 300;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}
.section-heading {
  font-family: 'DM Serif Display', serif;
  font-size: 18px;
  color: var(--primary);
  margin: 18px 0 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Stat / info badges ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--primary);
  margin: 3px;
}
.badge-green {
  background: #e8f5ed;
  border-color: #a8d5b5;
  color: #1a5c30;
}
.badge-amber {
  background: #fef4e0;
  border-color: #e8c84a;
  color: #7a5500;
}

/* ── Prediction result card ── */
.result-card {
  background: linear-gradient(135deg, #e8f5ed 0%, #f4fbf6 100%);
  border: 2px solid var(--primary-light);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin: 16px 0;
  box-shadow: 0 2px 12px rgba(46,139,87,0.12);
}
.result-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 6px;
}
.result-value {
  font-family: 'DM Serif Display', serif;
  font-size: 24px;
  color: var(--primary-dark);
  line-height: 1.2;
}
.result-conf {
  font-size: 14px;
  color: var(--primary-light);
  font-weight: 500;
  margin-top: 4px;
}
.result-card-warn {
  background: linear-gradient(135deg, #fdf3e6 0%, #fff9f0 100%);
  border-color: #e8a030;
}
.result-value-warn { color: #8a4400; }

/* ── Warning / disclaimer block ── */
.disclaimer {
  background: var(--warn-bg);
  border-left: 4px solid var(--warn-border);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 12px 16px;
  font-size: 13px;
  color: #6b5000;
  line-height: 1.6;
  margin-bottom: 20px;
}

/* ── Quick action list ── */
.qa-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}
.qa-item:last-child { border-bottom: none; }
.qa-icon {
  font-size: 20px;
  flex-shrink: 0;
  margin-top: 1px;
}
.qa-text { color: var(--text); line-height: 1.4; }
.qa-text b { color: var(--primary); }

/* ── Team member card ── */
.team-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  margin-bottom: 14px;
  box-shadow: var(--shadow);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}
.team-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  background: linear-gradient(to bottom, var(--primary-light), var(--accent));
  border-radius: 4px 0 0 4px;
}
.team-card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); }
.team-name {
  font-family: 'DM Serif Display', serif;
  font-size: 17px;
  color: var(--primary-dark);
  margin-bottom: 3px;
}
.team-role {
  font-size: 12px;
  color: var(--primary-light);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
}
.team-meta {
  font-size: 12.5px;
  color: var(--text-muted);
  line-height: 1.7;
}

/* ── Streamlit widget overrides ── */
.stTextInput > label,
.stTextArea > label,
.stSelectbox > label,
.stSlider > label {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text) !important;
  letter-spacing: 0.1px !important;
  margin-bottom: 4px !important;
}
.stTextInput > div > div > input,
.stTextArea > div > textarea {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface2) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  transition: var(--transition) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
  border-color: var(--primary-light) !important;
  background: var(--surface) !important;
  box-shadow: 0 0 0 3px rgba(46,139,87,0.12) !important;
}
div[data-baseweb="select"] > div {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface2) !important;
  font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--primary-light) !important;
  box-shadow: 0 0 0 3px rgba(46,139,87,0.12) !important;
}

/* Submit button */
.stFormSubmitButton > button,
.stButton > button {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 28px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  cursor: pointer !important;
  transition: var(--transition) !important;
  box-shadow: 0 2px 8px rgba(15,61,30,0.2) !important;
}
.stFormSubmitButton > button:hover,
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 16px rgba(15,61,30,0.28) !important;
  opacity: 0.95 !important;
}

/* Alerts */
.stSuccess > div {
  background: #e6f4ea !important;
  border-left: 4px solid var(--primary-light) !important;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
  color: var(--primary-dark) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stError > div {
  background: #fdecea !important;
  border-left: 4px solid var(--danger) !important;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stWarning > div {
  background: var(--warn-bg) !important;
  border-left: 4px solid var(--warn-border) !important;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stInfo > div {
  background: #e8f2fb !important;
  border-left: 4px solid #4a90d9 !important;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Spinner */
.stSpinner > div {
  border-top-color: var(--primary-light) !important;
}

/* Table */
.stTable table {
  border-collapse: collapse !important;
  width: 100% !important;
  border-radius: var(--radius-sm) !important;
  overflow: hidden !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13.5px !important;
}
.stTable thead th {
  background: var(--surface2) !important;
  color: var(--primary) !important;
  font-weight: 600 !important;
  padding: 10px 14px !important;
  border-bottom: 2px solid var(--border) !important;
}
.stTable tbody td {
  padding: 9px 14px !important;
  border-bottom: 1px solid var(--border) !important;
  color: var(--text) !important;
}
.stTable tbody tr:last-child td { border-bottom: none !important; }
.stTable tbody tr:hover td { background: var(--surface2) !important; }

/* Dividers */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 18px 0 !important; }

/* Checkbox */
.stCheckbox label {
  font-size: 14px !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
  font-weight: 500 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 12px !important;
  background: var(--surface2) !important;
  transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--primary-light) !important;
  background: #e8f5ed !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--primary) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
  background-color: var(--primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Expander */
.streamlit-expanderHeader {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  color: var(--primary) !important;
  font-weight: 500 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

/* ── Footer ── */
.app-footer {
  text-align: center;
  padding: 24px 0 12px;
  font-size: 12.5px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  margin-top: 32px;
  font-weight: 300;
  letter-spacing: 0.3px;
}
.app-footer strong { color: var(--primary); font-weight: 600; }

/* Divider with text */
.divider-text {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin: 20px 0;
}
.divider-text::before, .divider-text::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}
</style>
""", unsafe_allow_html=True)

# Brand bar
st.markdown("""
<div class="brand-bar">
  <div class="brand-icon">🌿</div>
  <div>
    <div class="brand-title">Betel Leaf Disease Detector</div>
    <div class="brand-sub">AI-powered leaf health analysis · EfficientNetV2 · Academic Research Project</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Model loading (robust)
# -----------------------
MODEL_BASENAMES = [
    "betel_leaf_efficientnetv2.keras",
    "betel_leaf_model.keras",
    "mobilenetv2_final.keras",
    "efficientnet_model.keras",
]

SEARCH_DIRS = [
    os.path.join("streamlit_betel_leaf_app", "models"),
    "models",
    ".",
]

@st.cache_resource
def load_model_from_paths():
    attempted = []
    load_errors = []
    for d in SEARCH_DIRS:
        for base in MODEL_BASENAMES:
            p = os.path.join(d, base)
            if os.path.exists(p):
                attempted.append(p)
                try:
                    m = tf.keras.models.load_model(p, compile=False)
                    return m, p, None
                except Exception as e:
                    tb = traceback.format_exc()
                    load_errors.append(f"{p}: {str(e)}")
            else:
                attempted.append(p)

    attempted_str = "\n".join(attempted)
    if load_errors:
        err_msg = "Tried these paths (files exist and some failed to load or were missing):\n"
        err_msg += attempted_str + "\n\nLoad errors:\n" + "\n".join(load_errors)
    else:
        err_msg = "Model not found. Tried paths:\n" + attempted_str
    return None, None, err_msg

model, model_path, model_err = load_model_from_paths()

CLASS_NAMES = [
    "Anthracnose_Green",
    "BacterialLeafSpot_Green",
    "Healthy_Green",
    "Healthy_Red",
]

CLASS_META = {
    "Anthracnose_Green":      {"icon": "🔴", "color": "#c0392b", "bg": "#fdecea"},
    "BacterialLeafSpot_Green":{"icon": "🟠", "color": "#d35400", "bg": "#fef0e6"},
    "Healthy_Green":          {"icon": "🟢", "color": "#1a5c30", "bg": "#e6f4ea"},
    "Healthy_Red":            {"icon": "🔵", "color": "#1a4a7a", "bg": "#e8f0fb"},
}

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

if model is not None:
    IMG_H, IMG_W = model.input_shape[1], model.input_shape[2]
else:
    IMG_H, IMG_W = 300, 300

# -----------------------
# Helper functions
# -----------------------
def preprocess_pil_image_advanced(pil_img, target_size=None):
    if target_size is None:
        target_size = (IMG_W, IMG_H)
    pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.resize(target_size, resample=Image.LANCZOS)
    arr = np.array(pil_img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def tta_predictions(model, pil_img, tta_transforms=None, target_size=None):
    if target_size is None:
        target_size = (IMG_W, IMG_H)
    if tta_transforms is None:
        tta_transforms = [
            lambda im: im,
            lambda im: ImageOps.mirror(im),
            lambda im: im.rotate(15, expand=False),
            lambda im: im.rotate(-15, expand=False),
        ]
    probs_list = []
    for tfm in tta_transforms:
        im2 = tfm(pil_img.copy())
        arr = preprocess_pil_image_advanced(im2, target_size)
        preds = model.predict(arr, verbose=0)[0]
        if abs(np.sum(preds) - 1.0) < 0.05:
            probs = preds
        else:
            probs = tf.nn.softmax(preds).numpy()
        probs_list.append(probs)
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg_probs

def predict_with_tta(model, pil_img, T=1.0):
    avg_probs = tta_predictions(model, pil_img, target_size=(IMG_W, IMG_H))
    if T is not None and T != 1.0:
        logits = np.log(avg_probs + 1e-12)
        logits = logits / T
        avg_probs = tf.nn.softmax(logits).numpy()
    idx = int(np.argmax(avg_probs))
    return idx, avg_probs

def file_size_human(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:3.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except:
        return "Unknown"

def render_prediction(idx, probs):
    """Render a polished prediction result block."""
    cls = CLASS_NAMES[idx]
    meta = CLASS_META[cls]
    conf = 100 * np.max(probs)
    is_healthy = "Healthy" in cls

    card_cls = "result-card" if is_healthy else "result-card result-card-warn"
    val_cls  = "result-value" if is_healthy else "result-value result-value-warn"

    st.markdown(f"""
    <div class="{card_cls}">
      <div class="result-label">Detected Condition</div>
      <div class="{val_cls}">{meta['icon']} {cls.replace('_', ' ')}</div>
      <div class="result-conf">Confidence: {conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability table with inline mini-bars
    st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
    rows = ""
    for i, (c, p) in enumerate(zip(CLASS_NAMES, probs * 100)):
        m = CLASS_META[c]
        bar_w = int(p)
        active = "font-weight:600;" if i == idx else ""
        rows += f"""
        <tr>
          <td style="padding:7px 12px;{active}font-size:13px;">{m['icon']} {c.replace('_',' ')}</td>
          <td style="padding:7px 12px;width:55%;">
            <div style="background:#edf2ef;border-radius:4px;height:8px;overflow:hidden;">
              <div style="background:{m['color']};width:{bar_w}%;height:100%;border-radius:4px;transition:width 0.4s;"></div>
            </div>
          </td>
          <td style="padding:7px 12px;font-size:13px;text-align:right;color:#5a7a62;{active}">{p:.2f}%</td>
        </tr>"""
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;border:1px solid var(--border);">
      <thead>
        <tr style="background:#eef6f1;">
          <th style="padding:9px 12px;text-align:left;font-size:11.5px;color:#1a5c30;text-transform:uppercase;letter-spacing:0.8px;">Class</th>
          <th style="padding:9px 12px;text-align:left;font-size:11.5px;color:#1a5c30;text-transform:uppercase;letter-spacing:0.8px;">Distribution</th>
          <th style="padding:9px 12px;text-align:right;font-size:11.5px;color:#1a5c30;text-transform:uppercase;letter-spacing:0.8px;">Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["🏠  Home", "🔍  Predict", "🌿  About Betel Leaf", "👥  About Us", "💬  Feedback"])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown('<div class="page-title">Betel Leaf Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Professional AI tool for detecting betel leaf diseases using deep learning.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-heading">📊 Dataset & Model</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-sm">Trained on <strong>~10,000 images</strong> across <strong>4 disease classes</strong> using EfficientNetV2.<br><br>'
            + " ".join([
                f'<span class="badge badge-green">{CLASS_META[c]["icon"]} {c.replace("_"," ")}</span>'
                for c in CLASS_NAMES
            ])
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-heading">⚙️ Model Status</div>', unsafe_allow_html=True)
        if model is not None and model_path:
            st.success(f"✅ Model loaded from `{model_path}` ({file_size_human(model_path)})")
        else:
            st.error("❌ Model not found or failed to load.")
            if model_err:
                with st.expander("Model loader details"):
                    st.code(model_err)
                st.info("Ensure model file is under one of: " + ", ".join(SEARCH_DIRS))
            st.markdown("Make sure `models/betel_leaf_efficientnetv2.keras` is present and Git LFS is enabled.")

        st.markdown('<div class="section-heading">🔗 Resources</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card-sm" style="font-size:13.5px;line-height:2.2;">
  📦 <a href="https://www.kaggle.com/datasets/achmadbauravindah/betel-leaf-disease-classification" target="_blank" style="color:var(--primary-light);">Kaggle Dataset</a><br>
  💻 <a href="https://github.com/Akash040917/streamlit_betel_leaf_app" target="_blank" style="color:var(--primary-light);">GitHub Repository</a><br>
  🧪 <a href="https://colab.research.google.com/drive/1N9yE22hXCalUVC_ir7nzaj9e7pgolTVu?usp=sharing" target="_blank" style="color:var(--primary-light);">Google Colab Notebook</a>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-heading">⚡ Quick Actions</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card-sm" style="padding:6px 16px;">
  <div class="qa-item">
    <div class="qa-icon">🧠</div>
    <div class="qa-text"><b>Run Disease Detection</b><br>Go to the <em>Predict</em> tab and upload or capture an image.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">🌿</div>
    <div class="qa-text"><b>Learn About Betel Leaves</b><br>Visit <em>About Betel Leaf</em> for plant details and varieties.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">👨‍💻</div>
    <div class="qa-text"><b>Meet the Team</b><br>Check the <em>About Us</em> tab to know our developers.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">💬</div>
    <div class="qa-text"><b>Share Your Thoughts</b><br>Use the <em>Feedback</em> tab to help us improve.</div>
  </div>
</div>
        """, unsafe_allow_html=True)

# -----------------------
# PREDICT
# -----------------------
with tabs[1]:
    st.markdown('<div class="page-title">Predict Betel Leaf Condition</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload an image or use your camera to detect diseases in real-time.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Academic Research Notice:</strong> This AI model is developed as part of an academic research project.
      Predictions are research-oriented and <strong>not standardized for large-scale agricultural deployment</strong>.
      Outputs are intended for <strong>educational and research purposes</strong> with scope for further enhancement.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📷 Camera Capture</div>', unsafe_allow_html=True)
    start_cam = st.checkbox("Enable Camera")
    if start_cam:
        captured = st.camera_input("Take a photo of the leaf")
        if captured:
            col_img, col_res = st.columns([1, 1], gap="large")
            with col_img:
                img = Image.open(captured)
                st.image(img, caption="Captured Image", use_container_width=True)
            with col_res:
                if model:
                    with st.spinner("Analysing with TTA…"):
                        idx, probs = predict_with_tta(model, img)
                    render_prediction(idx, probs)
                else:
                    st.warning("Model not available. See Home tab for details.")

    st.markdown('<div class="divider-text">or upload a file</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a betel leaf image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file:
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=350)
        with col_res:
            if model:
                with st.spinner("Analysing with TTA…"):
                    idx, probs = predict_with_tta(model, img)
                render_prediction(idx, probs)
            else:
                st.warning("Model not available. See Home tab for details.")

# -----------------------
# ABOUT BETEL LEAF
# -----------------------
with tabs[2]:
    st.markdown('<div class="page-title">About Piper betle (Betel Leaf)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Botany, traditional uses, and cultivar classification.</div>', unsafe_allow_html=True)

    col_img, col_text = st.columns([1, 2], gap="large")
    with col_img:
        st.image("streamlit_betel_leaf_app/images/betel.jpg", caption="Piper betle vine", use_container_width=True)
    with col_text:
        st.markdown("""
**Piper betle** is a perennial vine from the **Piperaceae** family, widely cultivated across South and Southeast Asia.
Heart-shaped leaves are revered in traditional medicine, culinary applications, and cultural rituals.

Key phytochemicals include **hydroxychavicol** and **eugenol**, which exhibit strong antimicrobial and antioxidant properties.
        """)
        st.markdown('<div class="section-heading">🌿 Varieties &amp; Classes</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card-sm">
  <div class="qa-item">
    <div class="qa-icon">🟢</div>
    <div class="qa-text"><b>Green Varieties</b> — Common in South India; softer texture and mild aroma.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">🔴</div>
    <div class="qa-text"><b>Red Varieties</b> — Thicker leaves, stronger flavour, preferred for traditional uses.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">📍</div>
    <div class="qa-text"><b>Regional Cultivars</b> — Includes <em>Banarasi Pan</em>, <em>Kalkatta Pan</em>, and GI-protected varieties of India.</div>
  </div>
  <div class="qa-item">
    <div class="qa-icon">🌱</div>
    <div class="qa-text"><b>Diversity</b> — Each type differs in taste, medicinal value, and essential oil content.</div>
  </div>
</div>
        """, unsafe_allow_html=True)

# -----------------------
# ABOUT US
# -----------------------
with tabs[3]:
    st.markdown('<div class="page-title">Our Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Final-year Mechatronics Engineering students — Rajalakshmi Engineering College.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-sm" style="margin-bottom:20px;">
    We are final-year <strong>Mechatronics Engineering</strong> students with a strong interest in AI and Deep Learning.
    This project uses a Kaggle betel leaf image dataset with <strong>4 disease classes</strong>
    and achieves approximately <strong>85% validation accuracy</strong>.
    Our objective is to build a practical, user-friendly AI system for automated betel leaf disease detection.
    </div>
    """, unsafe_allow_html=True)

    members = [
        {
            "name": "Abdul Rawoof M",
            "role": "Deep Learning Model Development & Image Preprocessing",
            "reg": "221201001",
            "email": "221201001@rajalakshmi.edu.in",
        },
        {
            "name": "Akash Raghuram R L",
            "role": "Application Development & System Integration",
            "reg": "221201004",
            "email": "221201004@rajalakshmi.edu.in",
        },
        {
            "name": "Sarath Kumar R",
            "role": "Dataset Preparation, Testing & Performance Evaluation",
            "reg": "221201048",
            "email": "221201048@rajalakshmi.edu.in",
        },
    ]

    cols = st.columns(3, gap="medium")
    for col, m in zip(cols, members):
        with col:
            st.markdown(f"""
            <div class="team-card">
              <div class="team-name">{m['name']}</div>
              <div class="team-role">{m['role']}</div>
              <div class="team-meta">
                🎓 Reg: {m['reg']}<br>
                ✉️ <a href="mailto:{m['email']}" style="color:var(--primary-light);text-decoration:none;">{m['email']}</a>
              </div>
            </div>
            """, unsafe_allow_html=True)

def save_to_gsheet(data):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    client = gspread.authorize(creds)
    sheet = client.open("Feedback").sheet1
    sheet.append_row(data)

# -----------------------
# FEEDBACK
# -----------------------
import re

with tabs[4]:

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    st.markdown('<div class="page-title">Share Your Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Help us improve the app — your thoughts matter.</div>', unsafe_allow_html=True)

    col_form, col_tip = st.columns([2, 1], gap="large")

    with col_form:

        # 🔥 MOVE SLIDER OUTSIDE FORM (THIS IS THE FIX)
        rating = st.slider(
            "Overall Rating",
            min_value=1,
            max_value=5,
            value=st.session_state.get("rating", 3),
            key="rating"
        )

        rating_labels = {
            1: ("😞", "Very Bad",  "#c0392b", "#fdecea"),
            2: ("😐", "Bad",       "#d35400", "#fef0e6"),
            3: ("🙂", "Neutral",   "#7a6000", "#fef9ec"),
            4: ("😊", "Good",      "#1a5c30", "#e6f4ea"),
            5: ("🤩", "Excellent", "#1a4a7a", "#e8f0fb"),
        }

        icon, label, color, bg = rating_labels[rating]

        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:7px;'
            f'background:{bg};border:1.5px solid {color};border-radius:20px;'
            f'padding:5px 14px;font-size:13px;font-weight:600;color:{color};margin-top:4px;margin-bottom:10px;">'
            f'{icon} {label} ({rating}/5)</div>',
            unsafe_allow_html=True
        )

        # 🔽 FORM STARTS HERE (WITHOUT SLIDER)
        with st.form("feedback_form", clear_on_submit=True):

            c1, c2 = st.columns(2)
            with c1:
                fname = st.text_input("Full Name", placeholder="Your name")
            with c2:
                femail = st.text_input("Email Address", placeholder="you@example.com")

            ftype = st.selectbox(
                "Feedback Type",
                ["Bug Report", "Feature Request", "Dataset Issue", "Other"]
            )

            fmsg = st.text_area(
                "Your Message",
                placeholder="Describe your experience, a bug you found, or a feature you'd love…",
                height=120
            )

            submit = st.form_submit_button("📨 Submit Feedback")

            if submit:

                if st.session_state.submitted:
                    st.warning("⚠️ Feedback already submitted. Refresh to submit again.")

                elif not fname or not femail or not fmsg:
                    st.warning("⚠️ Please fill in your name, email, and message.")

                elif not re.match(r"[^@]+@[^@]+\.[^@]+", femail):
                    st.warning("⚠️ Please enter a valid email address.")

                else:
                    try:
                        with st.spinner("Submitting feedback..."):
                            row = [
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                fname,
                                femail,
                                ftype,
                                rating,   # 🔥 still used here
                                fmsg,
                                model_path
                            ]
                            save_to_gsheet(row)

                        st.session_state.submitted = True
                        st.success("✅ Thank you! Your feedback has been recorded.")

                    except Exception as e:
                        st.error(f"Submission failed: {e}")

    with col_tip:
        st.markdown("""
        <div class="card-sm" style="margin-top:4px;">
          <div style="font-size:13px;font-weight:600;color:var(--primary);margin-bottom:10px;">💡 Tips for great feedback</div>
          <div style="font-size:13px;color:var(--text-muted);line-height:1.9;">
            ✔ Be specific about the issue<br>
            ✔ Include the image type if relevant<br>
            ✔ Mention the prediction result you saw<br>
            ✔ Suggest what you expected instead
          </div>
        </div>
        """, unsafe_allow_html=True)
            
# -----------------------
# Footer
# -----------------------
st.markdown(f"""
<div class="app-footer">
  <strong>ProjectASA2025</strong> &nbsp;·&nbsp; Built with Streamlit &amp; TensorFlow &nbsp;·&nbsp; © {time.strftime('%Y')}
</div>
""", unsafe_allow_html=True)
