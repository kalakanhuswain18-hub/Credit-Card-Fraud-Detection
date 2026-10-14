from pathlib import Path
import joblib
import pandas as pd 
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel,Field

# Root Folder
BASE_DIR=Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "lightgbm_froud_model.pkl"
METADATA_PATH=BASE_DIR/"models"/"model_metadata.json"

#Load Trained Model
model=joblib.load(MODEL_PATH)


import json

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts whether a transaction may be fraudulent.",
    version="1.0.0",
)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "model": metadata["model"],
    }


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Keep features in the exact order used during training
        input_df = pd.DataFrame(
            [transaction.model_dump()]
        )[model.feature_name_]

        probability = float(model.predict_proba(input_df)[0, 1])
        predicted_class = int(probability >= THRESHOLD)

        return {
            "fraud_probability": probability,
            "threshold": THRESHOLD,
            "predicted_class": predicted_class,
            "prediction": (
                "Fraud" if predicted_class == 1 else "Genuine"
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
