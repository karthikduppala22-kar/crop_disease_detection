import streamlit as st
from PIL import Image
from inference import load_infer, predict_image

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
)

st.title("🌱 Crop Disease Detection")
st.write("Upload a leaf image to detect crop disease.")

@st.cache_resource
def get_model():
    return load_infer()

infer, tf = get_model()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        label, confidence = predict_image(image, infer, tf)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence * 100:.2f}%")
