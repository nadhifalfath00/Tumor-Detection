import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image
import streamlit as st

# =========================
# KONFIGURASI UMUM
# =========================
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60  # %

CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

LAST_CONV_VGG = "block5_conv3"
LAST_CONV_RESNET = "conv5_block3_out"


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model_vgg = tf.keras.models.load_model(
        "model_VGG16_streamlit_ready.h5"
    )
    model_resnet = tf.keras.models.load_model(
        "model_ResNet50V2_streamlit_ready.h5"
    )
    return model_vgg, model_resnet


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# =========================
# PREDICTION
# =========================
def predict_image(model, image):
    img = preprocess_image(image)

    start_time = time.time()
    preds = model.predict(img, verbose=0)
    inference_time = time.time() - start_time

    if isinstance(preds, list):
        preds = preds[0]

    preds = preds[0]  # (4,)
    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index] * 100)

    return {
        "valid": confidence >= CONFIDENCE_THRESHOLD,
        "label": CLASS_NAMES[class_index],
        "confidence": confidence,
        "preds": preds,                     # ✅ PENTING
        "inference_time": inference_time
    }


# =========================
# GRAD-CAM
# =========================
def generate_gradcam(model, last_conv_layer, image, pred_index):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(image)
    gradcam = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return gradcam


# =========================
# DUAL MODEL OUTPUT
# =========================
def generate_dual_gradcam(image: Image.Image):
    model_vgg, model_resnet = load_models()

    vgg = predict_image(model_vgg, image)
    res = predict_image(model_resnet, image)

    output = {}

    # VGG16
    gradcam_vgg = generate_gradcam(
        model_vgg,
        LAST_CONV_VGG,
        image,
        np.argmax(vgg["preds"])
    )

    output["VGG16"] = {
        **vgg,
        "gradcam": gradcam_vgg
    }

    # ResNet50V2
    gradcam_res = generate_gradcam(
        model_resnet,
        LAST_CONV_RESNET,
        image,
        np.argmax(res["preds"])
    )

    output["ResNet50V2"] = {
        **res,
        "gradcam": gradcam_res
    }

    return output


# =========================
# BACKGROUND
# =========================
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_file}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
