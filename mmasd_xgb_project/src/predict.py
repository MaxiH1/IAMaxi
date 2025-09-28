import pandas as pd
import joblib

model = joblib.load("artifacts/model.pkl")
new_data = pd.read_csv("data/processed/new_samples.csv")

preds = model.predict(new_data)
print("Predicciones:", preds)
