"""FastAPI service: POST /predict -> bubble point pressure and Bo for a saturated oil.

Run locally (repo root):  uvicorn app.app:app --reload
Docs:                      http://127.0.0.1:8000/docs
"""
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pvt.predict import Predictor

app = FastAPI(
    title="PVT property predictor",
    description="Neural-network estimates of bubble point pressure (Pb) and oil formation volume "
                "factor (Bo) for saturated oils. Trained on 548 compiled literature records; "
                "see the README for accuracy and limits.",
    version="1.0.0",
)


class OilInputs(BaseModel):
    rs: float = Field(..., gt=0, le=10000, description="Solution GOR, scf/STB")
    gg: float = Field(..., gt=0.3, lt=4.0, description="Gas specific gravity (air = 1)")
    api: float = Field(..., gt=0, lt=80, description="Oil API gravity")
    temp: float = Field(..., gt=32, lt=500, description="Reservoir temperature, deg F")

    model_config = {"json_schema_extra": {"example": {"rs": 450, "gg": 0.75, "api": 32, "temp": 210}}}


class Prediction(BaseModel):
    pb_psia: float
    bo_rb_per_stb: float
    warnings: list[str]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@app.get("/health")
def health():
    get_predictor()
    return {"status": "ok", "models": ["pb", "bo"]}


@app.post("/predict", response_model=Prediction)
def predict(inputs: OilInputs):
    return get_predictor().predict(inputs.rs, inputs.gg, inputs.api, inputs.temp)
