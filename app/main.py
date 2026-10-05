"""FastAPI service: POST a request, get back an intent (or "out_of_scope")."""
import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline import predict_with_fallback

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "model.joblib")                       # load once at startup
THRESHOLD = json.loads((MODEL_DIR / "metrics.json").read_text())["threshold"]

app = FastAPI(title="Intent API", version="1.0")


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request) -> dict:
    proba = model.predict_proba([req.text])[0]
    intent = predict_with_fallback(model, [req.text], THRESHOLD)[0]
    return {"intent": intent, "confidence": round(float(proba.max()), 3)}