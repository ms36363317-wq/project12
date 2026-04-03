import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
st.write(os.path.exists("model.h5"))
st.write(os.path.getsize("model.h5"))
# ==============================

# Load Model
# ==============================
@st.cache_resource
MODEL_PATH = "model.keras"
MODEL_URL  = "https://drive.google.com/uc?id=13ZbZU6aYtHAs4cEeOwnDI_VRzTwZ0sUj"

def download_models():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    return load_model(model_path)

# 🔥 مهم جدًا (كان ناقص عندك)
model = load_model_cached()
st.write("Exists:", os.path.exists(model_path))

if os.path.exists(model_path):
    st.write("Size:", os.path.getsize(model_path))
else:
    st.write("❌ Model file not found yet")
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
    img = np.array(img) / 255.0
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
# Grad-CAM++
# ==============================
def gradcam(img, model):
    img = img.resize((300, 300))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    last_conv_layer = model.get_layer("top_conv")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    heatmap = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (300, 300))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# ==============================
# UI
# ==============================
st.title("👁️ Eye Disease AI")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original")

    # ✅ FIX
    pred, conf = predict(image, model)

    # ✅ FIX
    heatmap = gradcam(image, model)

    with col2:
        st.image(heatmap, caption="Grad-CAM++")

    with col3:
        st.image(image, caption=f"{pred}")

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf:.2f}")
