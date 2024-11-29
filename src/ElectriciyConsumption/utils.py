# import os
# import sys

# import numpy as np 
# import pandas as pd
# import dill
# import pickle
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

# from ElectriciyConsumption.logger.logger import logging

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)

#         os.makedirs(dir_path, exist_ok=True)

#         with open(file_path, "wb") as file_obj:
#             pickle.dump(obj, file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)


# def evaluate_models(x_train,y_train,x_test,y_test,models,param):
#      try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para = param[list(param.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(x_train,y_train)


#             model.set_params(**gs.best_params_)
#             model.fit(x_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(x_train)

#             y_test_pred = model.predict(x_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report
     
#      except Exception as e:
#          raise CustomException(e, sys)


# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)


import numpy as np
from datetime import datetime, timedelta
import joblib
import mysql.connector
from pathlib import Path

# Load the pre-trained model
model_reg_path = Path(r"src\models\regression_model.pkl")
model_clf_path = Path(r"src\models\Classifer.pkl")
try:
    model_reg = joblib.load(model_reg_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

try:
    model_clf = joblib.load(model_clf_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")




def fetch_historical_data():
    conn = mysql.connector.connect(
        host="localhost", user="root", password="password", database="electricity"
    )
    cursor = conn.cursor(dictionary=True)
    query = """SELECT *
                    FROM current_data
                    WHERE DateTime BETWEEN DATE_SUB(NOW(), INTERVAL 3 HOUR) AND NOW()
                    ORDER BY DateTime DESC
                    LIMIT 3;

"""
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data


def update_database(prediction,rolling_mean_3, rolling_std_3, DateTime):
    # Calculate the additional columns
    day_of_week = DateTime.weekday()  # Monday = 0, Sunday = 6
    hour_of_day = DateTime.hour  # Extract the hour of the timestamp
    is_weekend = 1 if day_of_week >= 5 else 0  # Weekend if Saturday (5) or Sunday (6)
    is_holiday = check_holiday(DateTime)  # Placeholder for a holiday checking function
    
    # Connect to the database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="electricity"
    )
    cursor = conn.cursor()
    
    # Update query with additional columns
    query = """
    INSERT INTO current_data (DateTime, electricity_demand, rolling_mean_3, rolling_std_3, day_of_week, hour_of_day, is_weekend, is_holiday)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    # Execute query
    cursor.execute(query, (DateTime, prediction, rolling_mean_3, rolling_std_3, day_of_week, hour_of_day, is_weekend, is_holiday))
    conn.commit()
    conn.close()

def check_holiday(timestamp):
    """
    Placeholder function to check if a given timestamp is a holiday.
    Replace this with actual holiday checking logic.
    """
    # Example: Return 1 if it's a fixed holiday (e.g., Jan 1st)
    if timestamp.strftime('%m-%d') in ['01-01', '12-25']:  # Example: New Year's and Christmas
        return 1
    return 0




def predict_24_hours(initial_features: dict, start_time: datetime, model_reg, model_clf) -> list:
    predictions = []
    historical_data = fetch_historical_data()

    rolling_mean_3 = np.mean([d['electricity_demand'] for d in historical_data[-3:]])
    rolling_std_3= np.std([d['electricity_demand'] for d in historical_data[-3:]])
    
    Previous_1_hour_demand = historical_data[-1]['electricity_demand']
    Previous_3_hour_demand = historical_data[-3]['electricity_demand']

    for hour in range(24):
        current_features = initial_features.copy()
        current_features.update({
            "Previous_1_hour_demand": Previous_1_hour_demand,
            "Previous_3_hour_demand": Previous_3_hour_demand,
            "rolling_mean_3": rolling_mean_3,
            "rolling_std_3": rolling_std_3
        })

        feature_array_reg = np.array([[current_features[key] for key in [
            "day_of_week", "hour_of_day", "is_weekend", "temperature", 
            "is_holiday", "solar_generation", "Previous_1_hour_demand", 
            "Previous_3_hour_demand", "rolling_mean_3", "rolling_std_3"]]])

        predicted_demand = model_reg.predict(feature_array_reg)[0]

        feature_array_clf = np.array([[
                            current_features["day_of_week"],
                            current_features["hour_of_day"],
                            current_features["is_weekend"],
                            current_features["temperature"],
                            current_features["is_holiday"],
                            current_features["solar_generation"],
                            predicted_demand  # Include predicted_demand here
                        ]])
        prediction_label = model_clf.predict(feature_array_clf)

        # Update rolling values
        rolling_mean_3 = (rolling_mean_3 * 3 + predicted_demand) / 4
        rolling_std_3 = np.std([historical_data[-1]['electricity_demand'], predicted_demand, historical_data[-3]['electricity_demand']])

        # Add prediction to database
        prediction_time = start_time + timedelta(hours=hour)
        update_database(predicted_demand, rolling_mean_3, rolling_std_3, prediction_time)

        # Append prediction result
        predictions.append({
            "timestamp": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_demand": predicted_demand,
            "prediction_label": int(prediction_label[0])
        })

        # Update previous demands
        Previous_1_hour_demand = predicted_demand
        Previous_3_hour_demand = Previous_1_hour_demand

    return predictions
