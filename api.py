from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model1 = joblib.load("ridge_model.pkl")
model2 = joblib.load("scaler.pkl")

class GoldInput(BaseModel):
    usd_price: float
    inr_rate: float = None
    mode: str = "Exact"

@app.post("/predict")
def predict_gold(data: GoldInput):
    if data.mode == "Exact" and data.inr_rate is not None:
        prediction = model1.predict([[data.usd_price, data.inr_rate]])[0]
    else:
        prediction = model2.predict([[data.usd_price]])[0]
    return {"gold_price_inr_per_gram": round(prediction, 2)}
