import os
import streamlit as st
import numpy as np
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# ==============================
# UI (Professional Version)
# ==============================

# Title
st.markdown('<div class="main-title">👁️ Eye Disease AI Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a retinal image to detect eye diseases using AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # ==============================
    # Prediction
    # ==============================
    pred, conf = predict(image, model)
    heatmap = gradcam(image, model)
    overlay = overlay_heatmap(image, heatmap)

    severity, color = DISEASE_INFO.get(pred, ("Unknown", "#999"))

    # ==============================
    # Images Section
    # ==============================
    st.markdown('<div class="section-header">🔍 Visual Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(heatmap, caption="Grad-CAM++", use_container_width=True)

    with col3:
        st.image(overlay, caption="AI Focus Area", use_container_width=True)

    # ==============================
    # Result Section
    # ==============================
    st.markdown('<div class="section-header">🧠 Diagnosis Result</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2,1])

    with col_left:
        st.markdown(
            f'<div class="disease-badge">{pred}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"<span style='color:{color}; font-weight:600;'>Severity: {severity}</span>",
            unsafe_allow_html=True
        )

        st.markdown('<div class="confidence-bar-label">Confidence</div>', unsafe_allow_html=True)
        st.progress(int(conf * 100))

    with col_right:
        st.metric(label="Confidence Score", value=f"{conf:.2f}")

    # ==============================
    # AI Report (Simple)
    # ==============================
    st.markdown('<div class="section-header">📄 AI Medical Insight</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="report-box">
        The AI model predicts <b>{pred}</b> with a confidence of <b>{conf:.2f}</b>.<br><br>
        The highlighted regions in the image indicate areas that contributed most to the decision.
        These regions may correspond to abnormal retinal patterns associated with the condition.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==============================
    # Warning
    # ==============================
    st.markdown(
        """
        <div class="warning-box">
        ⚠️ This AI result is for assistance only and should not replace professional medical diagnosis.
        Please consult an ophthalmologist for confirmation.
        </div>
        """,
        unsafe_allow_html=True
    )
# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model_cached():

    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found")
        st.stop()

    size = os.path.getsize(MODEL_PATH)
    st.write("Model size:", size)

    if size < 5_000_000:
        st.error("❌ Model corrupted")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except:
        st.error("❌ Failed to load model")
        st.stop()

    return model

model = load_model_cached()

# ==============================
# Classes
# ==============================
class_names = [
    'Diabetic Retinopathy',
    'Disc Edema',
    'Healthy',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

# ==============================
# Preprocess
# ==============================
def preprocess(img):
    img = img.resize((300, 300))
    img = np.array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ==============================
# Prediction
# ==============================
def predict(img, model):
    processed = preprocess(img)
    preds = model.predict(processed)
    idx = np.argmax(preds)
    return class_names[idx], float(np.max(preds))

# ==============================
# Grad-CAM
# ==============================
def gradcam(img, model):
    img = img.resize((300, 300))
    img = np.array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # 🔥 اختار layer قبل آخر pooling
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError("No Conv layer found")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # 🔥 لو gradients ضعيفة → نضخمها
    grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)

    cam = cam[0].numpy()

    # 🔥 مهم جدًا
    cam = np.maximum(cam, 0)

    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    # 🔥 Contrast boost
    cam = np.power(cam, 0.3)

    # resize
    cam = cv2.resize(cam, (300, 300))

    # color
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# ==============================
# Overlay
# ==============================
def overlay_heatmap(img, heatmap):
    img = img.resize((300, 300))
    img = np.array(img)

    # 🔥 قلل تأثير الأحمر
    overlay = cv2.addWeighted(img, 0.8, heatmap, 0.2, 0)
    return overlay

# ==============================
# UI
# ==============================
st.title("👁️ Eye Disease AI")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Prediction
    pred, conf = predict(image, model)
    heatmap = gradcam(image, model)
    overlay = overlay_heatmap(image, heatmap)

    # Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(heatmap, caption="Grad-CAM++")

    with col3:
        st.image(overlay, caption=f"Prediction: {pred}")

    # Results
    st.success(f"Prediction: {pred}")
    st.progress(int(conf * 100))
    st.info(f"Confidence: {conf:.2f}")
