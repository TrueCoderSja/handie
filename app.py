from fastapi import FastAPI, UploadFile, File
from fastai.vision.all import load_learner, PILImage
import tempfile
import os

app = FastAPI()

print("Loading Handie model...")
learn = load_learner("hands_detector.pkl")
print("Handie model loaded!")

@app.get("/")
def root():
    return {"message": "Handie API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        img = PILImage.create(temp_path)
        prediction, _, probabilities = learn.predict(img)

        return {
            "prediction": str(prediction),
            "confidence": float(probabilities.max())
        }

    finally:
        os.remove(temp_path)