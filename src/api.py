from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import json
from src.config import SCHEMA_PATH
from src.inference import predict_one

app = FastAPI(title="Used Car Price API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # vi bruker ikke cookies
)

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)
    

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        price = predict_one(req.features)
        return {"price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

