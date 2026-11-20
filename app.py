import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

art = joblib.load("outputs/model.joblib")
app = FastAPI(title="Fraud Detection API")


class Txn(BaseModel):
    features: dict[str, float]


@app.post("/predict")
def predict(t: Txn):
    missing = set(art["features"]) - set(t.features)
    if missing:
        raise HTTPException(422, f"Missing features: {sorted(missing)[:5]}...")
    X = pd.DataFrame([t.features])[art["features"]]
    p = float(art["model"].predict_proba(X)[0, 1])
    return {"fraud_probability": round(p, 4), "is_fraud": p >= art["threshold"]}