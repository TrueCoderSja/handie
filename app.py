import streamlit as st
from fastai.vision.all import load_learner, PILImage
import tempfile, os

st.title("Handie API")

@st.cache_resource
def get_model():
    return load_learner("hands_detector.pkl")

learn = get_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded.read())
        temp_path = temp.name

    try:
        img = PILImage.create(temp_path)
        prediction, _, probabilities = learn.predict(img)
        st.write(f"Prediction: **{prediction}**")
        st.write(f"Confidence: {float(probabilities.max()):.2%}")
    finally:
        os.remove(temp_path)