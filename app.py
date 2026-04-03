import os
import streamlit as st
import numpy as np
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# Config
# ==============================
MODEL_PATH = "model.h5"
FILE_ID = "11tjmQJITN0zHQ7x2wMPOF9L1JWnoZTxQ"

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

    # get last conv layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # 🔥 تحسين الحساب
    weights = tf.reduce_mean(grads, axis=(1, 2))

    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)
    cam = cam[0]

    heatmap = cam.numpy()

    # 🔥 تنظيف القيم
    heatmap = np.maximum(heatmap, 0)

    # normalize
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # 🔥 smoothing مهم جدًا
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # resize
    heatmap = cv2.resize(heatmap, (300, 300))

    # color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# ==============================
# Overlay
# ==============================
def overlay_heatmap(img, heatmap):
    img = img.resize((300, 300))
    img = np.array(img)

    # balance أفضل
    overlay = cv2.addWeighted(img, 0.75, heatmap, 0.25, 0)
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
