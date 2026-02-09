import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Image Classification")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_model.h5", compile=False)

model = load_model()

st.title("📸 MobileNetV2 Image Classification")

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    st.write(f"🧠 Class index: **{class_idx}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
