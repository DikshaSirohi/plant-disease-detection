import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_PATH = os.path.join("models", "improved_best.keras")
CLASS_NAMES_JSON = os.path.join("models", "class_names.json")
IMG_SIZE = (224, 224)


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


st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Leaf Disease Detection")
st.caption("Upload a leaf image — get a prediction and confidence score (38 classes).")

model = load_model()
class_names = load_class_names()

needs_div255 = not model_has_rescaling(model)

with st.expander("ℹ️ Model & Dataset Info"):
    st.write("- **Model:** EfficientNetB0 (fine-tuned)")
    st.write("- **Classes:** 38")
    st.write("- **Test accuracy (your evaluation):** ~99%")
    st.write("- **Note:** This tool is for academic/demo use.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess_image(img, needs_div255=needs_div255)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])

    raw_name = class_names[pred_idx] if class_names else f"Class #{pred_idx}"
    pred_name = pretty_label(raw_name)

    st.subheader("🔎 Prediction")
    tag_text, tag_kind = confidence_tag(pred_conf)
    getattr(st, tag_kind)(tag_text)

    st.write(f"### **{pred_name}**")
    st.progress(min(max(pred_conf, 0.0), 1.0))
    st.write(f"Confidence: **{pred_conf:.4f}**")

    if pred_conf < 0.6:
        st.warning(
            "⚠️ Low confidence detected.\n\n"
            "This image may be outside the training distribution.\n\n"
            "Try:\n"
            "- Cropping the image to include only the leaf\n"
            "- Using better lighting\n"
            "- Reducing background noise\n"
            "- Uploading a clearer, centered leaf image"
    )

    st.subheader("🏆 Top 5 Predictions")
    top5 = np.argsort(probs)[::-1][:5]
    top5_rows = []
    for i in top5:
        name = class_names[i] if class_names else f"Class #{i}"
        top5_rows.append((pretty_label(name), float(probs[i])))

    df_top5 = pd.DataFrame(top5_rows, columns=["Class", "Probability"])
    st.dataframe(df_top5, use_container_width=True, hide_index=True)

    # Bar chart
    fig = plt.figure()
    plt.bar(df_top5["Class"], df_top5["Probability"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Probability")
    plt.title("Top-5 probabilities")
    st.pyplot(fig)

    # Simple next-steps guidance (generic, not prescriptive)
    with st.expander("🧾 What you can do next (general guidance)"):
        st.write(
            "- Compare with multiple images (different angles/light).\n"
            "- If disease is suspected, isolate the plant and remove severely infected leaves.\n"
            "- Improve airflow; avoid overhead watering.\n"
            "- For actionable treatment, consult local agricultural extension services or an agronomist."
        )
        st.caption("Educational use only — not a substitute for professional advice.")

    # Download a small report
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction": raw_name,
        "prediction_pretty": pred_name,
        "confidence": pred_conf,
        "top5": [{"class": r[0], "probability": r[1]} for r in top5_rows],
    }
    st.download_button(
        "⬇️ Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json",
    )