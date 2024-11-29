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
encoder_path = Path(r"src\models\encoder.pkl")

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

try:
    encoder = joblib.load(encoder_path)
except Exception as e:
    raise RuntimeError(f"Error loading the encoder: {e}")

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
    # Transform input features for regression
        input_features_reg = np.array([
            input_data.day_of_week,
            input_data.hour_of_day,
            input_data.is_weekend,
            input_data.temperature,
            input_data.is_holiday,
            input_data.solar_generation,
            input_data.Previous_1_hour_demand,
            input_data.Previous_3_hour_demand,
            input_data.rolling_mean_3,
            input_data.rolling_std_3
        ]).reshape(1, -1)

        input_features_clf = np.array([
            input_data.day_of_week,
            input_data.hour_of_day,
            input_data.is_weekend,
            input_data.temperature,
            input_data.is_holiday,
            input_data.solar_generation,
        ]).reshape(1, -1)

        # Scale input features for regression
        input_features_reg = scaler.transform(input_features_reg)
        print("Input features for reg:")
        print(input_features_reg)

        # Predict demand with regression model
        prediction_reg = model_reg.predict(input_features_reg)
        print("Prediction (demand):")
        print(prediction_reg)

        # Append regression prediction to classification features
        input_features_clf = np.append(input_features_clf, prediction_reg.reshape(1, -1), axis=1)
        print("Input features for clf:\n")
        print(input_features_clf)

        # Predict label with classification model
        prediction_clf = model_clf.predict(input_features_clf)
        label = encoder.inverse_transform(prediction_clf.ravel())  # Flatten prediction

        # Return predictions
        return {
            "prediction_demand": float(prediction_reg[0]),  # Convert to float
            "prediction_label": label.tolist()             # Convert to list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
