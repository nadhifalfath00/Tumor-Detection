import streamlit as st

def show_sidebar():
    with st.sidebar:

        st.image("assets/logo.png", width=120)


        menu = st.radio(
            "Menu",
            ["Home", "Deteksi", "Model", "About"]
        )

        model_info = None

        # Kalau menu Model diklik → munculkan dropdown
        if menu == "Model":
            st.markdown("### 📚 Informasi Model")

            model_info = st.selectbox(
                "Pilih Arsitektur",
                ["ResNet50V2", "VGG16"]
            )

    return menu, model_info
