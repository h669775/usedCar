import joblib, pandas as pd
from src.config import MODEL_PATH

_bundle = joblib.load(MODEL_PATH)
PIPE = _bundle['pipeline']
FEATURES = _bundle['features']
TARGET = _bundle['target']



def predict_one(feat_dict):
    # Sikre at alle features er med (mangler → None)
    row = {k: feat_dict.get(k, None) for k in FEATURES}
    X = pd.DataFrame([row])
    yhat = PIPE.predict(X)[0]
    return float(yhat)
