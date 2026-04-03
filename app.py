import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model_cached():

    model_path = "model.h5"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=13ZbZU6aYtHAs4cEeOwnDI_VRzTwZ0sUj"
        gdown.download(url, model_path, quiet=False)

    return load_model(model_path)
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
def predict(img):
    processed = preprocess(img)
    preds = model.predict(processed)
    idx = np.argmax(preds)
    return class_names[idx], float(np.max(preds))

# ==============================
# Grad-CAM++
# ==============================
def gradcam(img):
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
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    heatmap = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    heatmap = np.maximum(heatmap, 0)
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

    pred, conf = predict(image)

    heatmap = gradcam(image)

    with col2:
        st.image(heatmap, caption="Grad-CAM++")

    with col3:
        st.image(image, caption=f"{pred}")

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf:.2f}")
