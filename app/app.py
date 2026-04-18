import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
 
MODEL_PATH       = os.path.join("models", "improved_best.keras")
CLASS_NAMES_JSON = os.path.join("models", "class_names.json")
IMG_SIZE         = (224, 224)
 
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
 
def pretty_label(s: str) -> str:
    return s.replace("___", " — ").replace("_", " ")
 
 
def model_has_rescaling(model: tf.keras.Model) -> bool:
    def walk(m):
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            if isinstance(layer, tf.keras.Model) and walk(layer):
                return True
        return False
    return walk(model)
 
 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)
 
 
def load_class_names():
    if os.path.exists(CLASS_NAMES_JSON):
        with open(CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
 
 
def preprocess_image(img: Image.Image, needs_div255: bool) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    if needs_div255:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)
 
 
def confidence_tag(conf: float) -> tuple[str, str]:
    if conf >= 0.90:
        return "High confidence ✅", "success"
    if conf >= 0.70:
        return "Medium confidence ⚠️", "warning"
    return "Low confidence ❓", "error"
 
 
def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()
 
 
# ─────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────
 
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
)
 
st.markdown("""
<style>
[data-testid="stHeader"] { background: transparent; }
 
.hero {
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2rem; margin: 0 0 0.4rem; color: white; }
.hero p  { font-size: 1rem; opacity: 0.9; margin: 0; color: white; }
 
.result-card {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(128,128,128,0.25);
    margin: 1rem 0;
}
.result-card .label {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: inherit;
}
.result-card .sublabel {
    font-size: 0.9rem;
    opacity: 0.6;
}
 
.badge-healthy { color: #1a4731; background: #d8f3dc; padding: 3px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.badge-disease { color: #7b1a26; background: #fde0e3; padding: 3px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
 
[data-testid="stFileUploader"] {
    border: 2px dashed #52b788 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────
 
st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image — get an instant AI-powered diagnosis across 38 plant classes</p>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
 
model        = load_model()
class_names  = load_class_names()
needs_div255 = not model_has_rescaling(model)
 
 
# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
 
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    | | |
    |---|---|
    | **Model** | EfficientNetB0 |
    | **Classes** | 38 |
    | **Accuracy** | ~99% |
    | **Input size** | 224 × 224 |
    """)
    st.caption("Fine-tuned on New Plant Diseases Dataset (Kaggle).")
    st.divider()
    st.markdown("### 📌 Tips for best results")
    st.markdown("""
    - Use a **clear, well-lit** photo
    - **Crop** to show only the leaf
    - Avoid heavy shadows or blur
    - Single leaf works better than whole plant
    """)
    st.divider()
    st.caption("For academic/demo use only — not a substitute for professional agricultural advice.")
 
 
# ─────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────
 
uploaded = st.file_uploader(
    "📂 Upload a leaf image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)
 
if uploaded is not None:
    img = Image.open(uploaded)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)
 
    # ── Predict ──
    x     = preprocess_image(img, needs_div255=needs_div255)
    probs = model.predict(x, verbose=0)[0]
 
    pred_idx  = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    raw_name  = class_names[pred_idx] if class_names else f"Class #{pred_idx}"
    pred_name = pretty_label(raw_name)
 
    tag_text, tag_kind = confidence_tag(pred_conf)
    status_badge = (
        '<span class="badge-healthy">🌱 Healthy</span>'
        if is_healthy(pred_name)
        else '<span class="badge-disease">⚠️ Disease Detected</span>'
    )
 
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="sublabel">Prediction</div>
            <div class="label">{pred_name}</div>
            <br>
            {status_badge}
            <br><br>
            <div class="sublabel">Confidence</div>
            <div style="font-size:1.6rem; font-weight:700;">{pred_conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        getattr(st, tag_kind)(tag_text)
 
    # Confidence progress bar
    st.progress(min(max(pred_conf, 0.0), 1.0))
 
    # Low-confidence warning
    if pred_conf < 0.6:
        st.warning(
            "⚠️ Low confidence — the image may be outside the training distribution.\n\n"
            "Try cropping to only the leaf, using better lighting, or reducing background clutter."
        )
 
    st.divider()
 
    # ── Top 5 ──
    st.markdown("#### 🏆 Top 5 Predictions")
    top5_indices = np.argsort(probs)[::-1][:5]
    top5_rows = [
        (pretty_label(class_names[i] if class_names else f"Class #{i}"), float(probs[i]))
        for i in top5_indices
    ]
    df_top5 = pd.DataFrame(top5_rows, columns=["Class", "Probability"])
 
    # Horizontal bar chart — transparent background so it fits any theme
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#2d6a4f" if i == 0 else "#95d5b2" for i in range(5)]
    ax.barh(df_top5["Class"][::-1], df_top5["Probability"][::-1], color=colors[::-1])
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig)
 
    with st.expander("📋 Show full table"):
        st.dataframe(df_top5, use_container_width=True, hide_index=True)
 
    st.divider()
 
    # ── Guidance ──
    with st.expander("🧾 What to do next"):
        st.markdown("""
        - Compare predictions using multiple photos (different angles/lighting).
        - If a disease is suspected, isolate the plant and remove badly infected leaves.
        - Improve airflow around the plant and avoid overhead watering.
        - For a proper treatment plan, consult a local agronomist or agricultural extension service.
        """)
        st.caption("Educational use only — not a substitute for professional advice.")
 
    # ── Download report ──
    report = {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "prediction":        raw_name,
        "prediction_pretty": pred_name,
        "confidence":        pred_conf,
        "top5": [{"class": r[0], "probability": r[1]} for r in top5_rows],
    }
    st.download_button(
        "⬇️ Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json",
    )
 
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; opacity: 0.4;">
        <div style="font-size: 4rem;">🍃</div>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Upload a leaf image above to get started</p>
    </div>
    """, unsafe_allow_html=True)