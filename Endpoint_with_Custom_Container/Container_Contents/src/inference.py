
import os

import joblib
import pandas as pd
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


# def load_model(model_dir: str) -> XGBClassifier:
def load_model(model_dir: str) -> DecisionTreeClassifier:
    """
    Load the model from the specified directory.
    """
    return joblib.load(os.path.join(model_dir, "model.joblib"))


# def predict(body: dict, model: XGBClassifier) -> dict:
def predict(body: dict, model: DecisionTreeClassifier) -> dict:
    """
    Generate predictions for the incoming request using the model.
    """
    features = pd.DataFrame.from_records(body["features"])
    predictions = model.predict(features).tolist()
    return {"predictions": predictions}