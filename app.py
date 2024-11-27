from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import uvicorn
from pathlib import Path

import sys
print("Python executable:", sys.executable)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get('/hello')
def hello():
    return {"message": "Endpoint is working"}

# Corrected model path
model_path = Path(r"D:/Projects/ElectriciyConsumption/src/models/regression_model.pkl")

# Load the model
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

class PredictionInput(BaseModel):
    day_of_week: int
    hour_of_day: float
    is_weekend: int
    temperature: float
    is_holiday: int
    solar_generation: float
    Previous_1_hour_demand: float
    Previous_3_hour_demand: float
    rolling_mean_3: float
    rolling_std_3: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Map correct field names to model's expected feature names
        input_features = np.array([
            input_data.day_of_week,
            input_data.hour_of_day,
            input_data.is_weekend,
            input_data.temperature,
            input_data.is_holiday,
            input_data.solar_generation,
            input_data.Previous_1_hour_demand,  # Correct name mapped to typo
            input_data.Previous_3_hour_demand,  # Correct name mapped to typo
            input_data.rolling_mean_3,
            input_data.rolling_std_3
        ]).reshape(1, -1)

        prediction = model.predict(input_features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
