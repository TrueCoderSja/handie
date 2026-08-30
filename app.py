from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner, PILImage
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

learn = load_learner("hands_detector.pkl")


@app.get("/")
def root():
    return {"status": "Handie is alive"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img = PILImage.create(BytesIO(image_data))

    prediction, _, probabilities = learn.predict(img)

    return {
        "prediction": str(prediction),
        "confidence": float(probabilities.max())
    }