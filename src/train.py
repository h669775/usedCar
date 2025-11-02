import json, joblib
import pandas as pd
import inspect
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.config import RAW_TRAIN, TARGET_CANDIDATES, MODEL_PATH, SCHEMA_PATH, METRICS_PATH, RANDOM_STATE, TEST_SIZE
from src.utils import infer_columns, save_schema

# 1) Load
df = pd.read_csv(RAW_TRAIN)

# 2) Finn target
target = next((t for t in TARGET_CANDIDATES if t in df.columns), None)
if target is None:
    raise ValueError(f"Fant ikke target blant {TARGET_CANDIDATES}. Kolonner: {list(df.columns)}")

# (valgfritt) fjern åpenbare ID-kolonner hvis finnes
id_like = [c for c in df.columns if c.lower() in {'id', 'car_id'}]

# 3) Split
train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# 4) Kolonner
num_cols, cat_cols = infer_columns(train_df, target, id_like_cols=id_like)
feature_cols = num_cols + cat_cols

# 5) Preprocessor
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    ohe_kwargs["sparse_output"] = False   # ny sklearn
else:
    ohe_kwargs["sparse"] = False          # gammel sklearn

pre = ColumnTransformer(
    transformers=[
        # StandardScaler på numeriske (dense)
        ('num', StandardScaler(), num_cols),
        # OneHotEncoder på kategoriske (dense)
        ('cat', OneHotEncoder(**ohe_kwargs), cat_cols),
    ],
    remainder='drop'  # output blir dense fordi alle trafoer gir dense
)

# 6) Modell (rask og god på tabell)
reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

pipe = Pipeline([
    ('pre', pre),
    ('reg', reg)
])

# 7) Fit
X_train, y_train = train_df[feature_cols], train_df[target]
X_val,   y_val   = val_df[feature_cols],   val_df[target]
pipe.fit(X_train, y_train)

# 8) Eval
pred = pipe.predict(X_val)
try:
    rmse = mean_squared_error(y_val, pred, squared=False)   # nyere sklearn
except TypeError:
    rmse = mean_squared_error(y_val, pred) ** 0.5           # eldre sklearn

mae = mean_absolute_error(y_val, pred)
r2  = r2_score(y_val, pred)

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_PATH, 'w') as f:
    json.dump({'rmse': rmse, 'mae': mae, 'r2': r2}, f, indent=2)

# 9) Lagre artefakter
save_schema(SCHEMA_PATH, feature_cols, {c: str(df[c].dtype) for c in feature_cols})
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({'pipeline': pipe, 'target': target, 'features': feature_cols}, MODEL_PATH)

print(f"Saved → {MODEL_PATH}\nRMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")
