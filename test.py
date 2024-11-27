import joblib

try:
    model = joblib.load('D:\Projects\ElectriciyConsumption/artifacts\models/regression_model.pkl')
    print(type(model))
    print("Model loaded Sucessfully")
except Exception as e:
    print(e)


print(model.feature_names_)
