import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

with open("configs/xgb_baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["dataset"]["path"])
X = df.drop(columns=[config["dataset"]["target"]])
y = df[config["dataset"]["target"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(**config["model"]["params"])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

joblib.dump(model, config["output"]["model_path"])
pd.DataFrame({"metric": ["accuracy"], "value": [acc]}).to_csv(config["output"]["metrics_path"], index=False)

print(f"Entrenamiento completado. Accuracy: {acc:.3f}")
