from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
package = joblib.load(MODEL_PATH)
model = package["model"]

app = FastAPI(title="Iris Flower Prediction API")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def home():
    return {"message": "Iris ML API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        "predicted_class": str(prediction),
        "confidence": round(float(max(probabilities)), 4)
    }
