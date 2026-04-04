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
IMG_SIZE = (300, 300)

# 🔥 مهم: غيرها لو layer مختلفة
LAST_CONV_LAYER = "top_conv"

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
# Grad-CAM++
# ==============================
def make_gradcam_plus_plus(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    grads_squared = tf.square(grads)
    grads_cubed = grads_squared * grads

    denominator = 2 * grads_squared + tf.reduce_sum(
        conv_outputs * grads_cubed, axis=(0, 1), keepdims=True
    )

    denominator = tf.where(denominator != 0, denominator, tf.ones_like(denominator))

    alphas = grads_squared / denominator
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))

    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), int(pred_index)

# ==============================
# Generate Grad-CAM++
# ==============================
def generate_gradcam_plus_plus(image):

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    heatmap, pred_idx = make_gradcam_plus_plus(
        img_array, model, LAST_CONV_LAYER
    )

    pred_label = class_names[pred_idx]

    original = np.array(img_resized)

    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 5)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    return original, heatmap, overlay, pred_label

# ==============================
# UI
# ==============================
st.title("👁️ Eye Disease AI")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    original, heatmap, overlay, pred = generate_gradcam_plus_plus(image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)

    with col2:
        st.image(heatmap, caption="Grad-CAM++", use_container_width=True)

    with col3:
        st.image(overlay, caption=f"Prediction: {pred}", use_container_width=True)
