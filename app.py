from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import uvicorn
from pathlib import Path
from datetime import datetime
from src.ElectriciyConsumption.utils import predict_24_hours,fetch_historical_data,update_database
from sklearn.preprocessing import MinMaxScaler


import sys
print("Python executable:", sys.executable)
scaler = MinMaxScaler()


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
model_reg_path = Path(r"src/models/regression_model.pkl")
model_clf_path = Path(r"src/models/Classifer.pkl")
scaler_path = Path(r"src\models\scaler.pkl")

# Load the model
try:
    model_reg = joblib.load(model_reg_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    model_clf = joblib.load(model_clf_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Error loading the scaler: {e}")

class PredictionInput_reg(BaseModel):
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


class PredictionInput_reg_24(BaseModel):
    day_of_week: int
    hour_of_day: float
    is_weekend: int
    temperature: float
    is_holiday: int
    solar_generation: float


class PredictionInput_clf(BaseModel):
    day_of_week : int
    hour_of_day : int
    is_weekend: int
    temperature: float
    is_holiday : int
    solar_generation: float
    electricity_demand: float



@app.post("/predict")
def predict(input_data: PredictionInput_reg):
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
        input_features = scaler.transform(input_features)
        print("Input features are:\n")
        print(input_features)

        prediction = model_reg.predict(input_features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictionOutput_reg(BaseModel):
    predictions: list[dict]

# @app.post("/predict_24", response_model=PredictionOutput_reg)
# async def predict_24(input_data: PredictionInput_reg_24):
#     """
#     Endpoint to predict 24-hour electricity demand.
#     """
#     try:
#         # Get current time as the starting point
#         start_time = datetime.now()

#         # Call prediction function
#         predictions = predict_24_hours(input_data.dict(), start_time,model_reg,model_clf)

#         return {"predictions": predictions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict_label")
async def predict_label(input_data:PredictionInput_clf):
    input_features = np.array([
        input_data.day_of_week,
        input_data.hour_of_day,
        input_data.is_weekend,
        input_data.temperature,
        input_data.is_holiday,
        input_data.solar_generation,
        input_data.electricity_demand
    ]).reshape(1, -1)

    try:
        prediction = model_clf.predict(input_features)

        return {"prediction_label": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
