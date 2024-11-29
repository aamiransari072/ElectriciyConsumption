# import joblib

# try:
#     model = joblib.load('D:\Projects\ElectriciyConsumption/artifacts\models/regression_model.pkl')
#     print(type(model))
#     print("Model loaded Sucessfully")
# except Exception as e:
#     print(e)


# print(model.feature_names_)


# from src.ElectriciyConsumption.utils import predict_24_hours


# from datetime import datetime

# di = {}

# x = predict_24_hours(di,datetime.now())


# import joblib
# import numpy as np

# # Load the model
# model = joblib.load('src\\models\\Classifer.pkl')

# # Check feature names
# print("Feature names:", model.feature_names_)

# # Prepare input features
# features = [2, 14, 0, 25.3, 0, 5.6, 400.7]
# features = np.array(features).reshape(1, -1)  # Reshape to 2D array

# # Make prediction
# prediction = model.predict(features)

# # Output results
# print("Prediction:", int(prediction[0]))  # Convert to int for clarity
# print("Prediction Type:", type(prediction))



from src.ElectriciyConsumption.utils import fetch_historical_data , predict_24_hours
from pathlib import Path
import joblib
from datetime import datetime


model_reg_path = Path(r"src\models\regression_model.pkl")
model_clf_path = Path(r"src\models\Classifer.pkl")

# Load the model
try:
    model_reg = joblib.load(model_reg_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    model_clf = joblib.load(model_clf_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")



input_data = {
    "day_of_week": 2,  # Tuesday
    "hour_of_day": 14,  # 2 PM
    "is_weekend": 0,  # False (Not a weekend)
    "temperature": 27.5,  # Degrees Celsius
    "is_holiday": 0,  # False (Not a holiday)
    "solar_generation": 12.3  # kWh (solar generation value)
}

now = datetime.now()

reponse = predict_24_hours(input_data,now,model_reg,model_clf)
print(reponse)


# response = fetch_historical_data()
# print(response)








