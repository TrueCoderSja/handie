import streamlit as st
from fastai.vision.all import load_learner, PILImage
import tempfile
import os

st.set_page_config(page_title="Handie", page_icon="🖐️")
st.title("🖐️ Handie")
st.caption("Upload a photo or take one with your webcam to classify a hand gesture.")


@st.cache_resource
def get_model():
    return load_learner("hands_detector.pkl")


with st.spinner("Loading model..."):
    learn = get_model()


def run_prediction(image_bytes: bytes, suffix: str = ".jpg"):
    """Save bytes to a temp file, run the fastai learner, return (prediction, confidence)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(image_bytes)
        temp_path = temp.name

    try:
        img = PILImage.create(temp_path)
        prediction, _, probabilities = learn.predict(img)
        return str(prediction), float(probabilities.max())
    finally:
        os.remove(temp_path)


# --- Input mode selector ---
mode = st.radio(
    "Choose image source",
    ["📁 Upload a file", "📷 Use webcam"],
    horizontal=True,
)

image_bytes = None
suffix = ".jpg"

if mode == "📁 Upload a file":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_bytes = uploaded.read()
        suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

else:  # Webcam
    st.info("Grant camera access when your browser prompts you, then click the capture button.")
    camera_photo = st.camera_input("Take a picture")
    if camera_photo is not None:
        image_bytes = camera_photo.getvalue()
        suffix = ".jpg"
        # st.camera_input already previews the captured photo, no need to re-display

# --- Run prediction ---
if image_bytes is not None:
    if st.button("Predict", type="primary"):
        with st.spinner("Running prediction..."):
            prediction, confidence = run_prediction(image_bytes, suffix)

        st.success(f"Prediction: **{prediction}**")
        st.metric("Confidence", f"{confidence:.2%}")
else:
    st.write("👆 Provide an image to get a prediction.")