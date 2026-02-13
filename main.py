import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from util import set_background, generate_dual_gradcam
from sidebar import show_sidebar

# =========================
# BASIC SETUP
# =========================
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

set_background("bg.png")

# 👉 SIDEBAR CONTROL
menu, model_info = show_sidebar()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title-text {
    font-size: 30px;
    font-weight: 800;
    text-align: center;
}
.subtitle-text {
    text-align: center;
    font-size: 16px;
    color: #444;
}
.result-box {
    padding: 25px;
    border-radius: 15px;
    background: rgba(255,255,255,0.95);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title-text">🧠 Brain MRI Tumor Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Aplikasi berbasis <b>Deep Learning (CNN)</b> untuk klasifikasi citra MRI otak</div>',
    unsafe_allow_html=True
)
st.divider()

if menu == "Home":
    st.subheader("Selamat Datang di Sistem Deteksi Tumor Otak Berbasis AI")

    st.write("""
    Aplikasi ini dirancang untuk membantu proses analisis citra MRI otak 
    menggunakan teknologi **Deep Learning Convolutional Neural Network (CNN)**.

    Sistem mampu:
    - Mengklasifikasikan jenis tumor otak
    - Membandingkan performa model VGG16 dan ResNet50V2
    - Menampilkan waktu inferensi masing-masing model
    - Menyediakan visualisasi Grad-CAM untuk interpretasi hasil

    👉 Gunakan menu **Deteksi** pada sidebar untuk memulai analisis citra MRI.
    """)

    st.info("Sistem ini dibuat untuk kepentingan akademik dan penelitian.")
    st.stop()


# =========================
# ABOUT PAGE
# =========================
if menu == "About":
    st.title("ℹ️ About This System")
    st.write("""
    Sistem ini dikembangkan untuk penelitian klasifikasi tumor otak menggunakan Deep Learning.

    Model yang dibandingkan:
    - VGG16
    - ResNet50V2

    Sistem juga menampilkan Grad-CAM untuk interpretasi area fokus model.
    """)
    st.stop()

# =========================
# MODEL INFO PAGE
# =========================
elif menu == "Model":

    st.title("📘 Penjelasan Arsitektur Model")

    if model_info == "ResNet50V2":
        st.subheader("ResNet50V2")
        st.write("""
        R ResNet50V2 (Residual Network) merupakan arsitektur Convolutional Neural Network (CNN) 
        yang menggunakan konsep *residual learning* melalui shortcut connection.
        """)
        
        st.markdown("### 🎯 Alasan Menggunakan ResNet50V2")
        st.write("""
        - Mengatasi masalah **vanishing gradient** pada jaringan yang dalam.
        - Mampu mengekstraksi fitur kompleks dari citra MRI otak.
        - Arsitektur dalam (50 layer) membuat model lebih akurat mengenali pola tumor.
        - Cocok untuk **medical image classification** karena mampu menangkap detail tekstur halus.
        - Versi V2 memiliki peningkatan stabilitas training dibanding versi pertama.
        """)

    elif model_info == "VGG16":
        st.subheader("VGG16")
        st.write("""
        VGG16 adalah arsitektur CNN dengan 16 layer yang dikembangkan oleh Visual Geometry Group (Oxford).
        Model ini terkenal dengan struktur sederhana dan konsisten menggunakan filter 3x3.
        """)

        st.markdown("### 🎯 Alasan Menggunakan VGG16")
        st.write("""
        - Struktur arsitektur **sederhana dan stabil** untuk klasifikasi citra.
        - Efektif dalam ekstraksi fitur dasar seperti tepi, tekstur, dan bentuk.
        - Sering dijadikan **baseline model** dalam penelitian klasifikasi citra medis.
        - Membutuhkan komputasi lebih ringan dibanding model modern yang sangat dalam.
        - Cocok sebagai pembanding performa terhadap model ResNet50V2.
        """)
    st.stop()
# =========================
# DETECTION PAGE
# =========================
st.header("📤 Upload MRI Image")

file = st.file_uploader(
    "Unggah citra MRI otak (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if file:
    image = Image.open(file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image (Original)",
        use_container_width=True
    )

    st.write(
        f"**Nama File:** {file.name}  \n"
        f"**Resolusi Asli:** {image.size[0]} x {image.size[1]} pixel  \n"
        f"**Resolusi Input Model:** 224 x 224 pixel"
    )

    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("🧠 Analyzing MRI image..."):
            st.session_state.result = generate_dual_gradcam(image)

    # =========================
# RESULT
# =========================
if "result" in st.session_state:

    result = st.session_state.result

    # VALIDATION WARNING
    if (not result["VGG16"]["valid"] or not result["ResNet50V2"]["valid"]):
        st.error("⚠️ Citra kemungkinan bukan MRI otak yang valid.")

    # =========================
    # PREDICTION SUMMARY
    # =========================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 VGG16")
        st.success(f"Prediction: {result['VGG16']['label']}")
        st.info(f"Confidence: {result['VGG16']['confidence']:.2f}%")

    with col2:
        st.subheader("🧠 ResNet50V2")
        st.success(f"Prediction: {result['ResNet50V2']['label']}")
        st.info(f"Confidence: {result['ResNet50V2']['confidence']:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # INFERENCE TIME
    # =========================
    st.divider()
    st.subheader("⏱️ Inference Time Comparison")
    col1.metric("VGG16", f"{result['VGG16']['inference_time']:.4f} detik")
    col2.metric("ResNet50V2", f"{result['ResNet50V2']['inference_time']:.4f} detik")

    # =========================
    # BAR CHART
    # =========================
    st.divider()
    st.subheader("📊 Probability Comparison per Class")

    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    vgg_probs = result["VGG16"]["preds"] * 100
    resnet_probs = result["ResNet50V2"]["preds"] * 100

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, vgg_probs, width, label="VGG16")
    ax.bar(x + width/2, resnet_probs, width, label="ResNet50V2")
    ax.set_ylabel("Probability (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)

    st.pyplot(fig)

    # =========================
    # GRAD-CAM
    # =========================
    st.divider()
    st.subheader("🔥 Grad-CAM Visualization")

    col1, col2 = st.columns(2)
    col1.image(result["VGG16"]["gradcam"], caption="VGG16 Grad-CAM", use_container_width=True)
    col2.image(result["ResNet50V2"]["gradcam"], caption="ResNet50V2 Grad-CAM", use_container_width=True)

    st.warning(
        "⚠️ **Disclaimer**  \n"
        "Aplikasi ini dikembangkan untuk kepentingan akademik. "
        "Hasil prediksi bukan diagnosis medis resmi."
    )