import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# CONFIG (edit here)
# ======================
MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_rf_model.joblib")
DATA_PATH  = os.getenv("DATA_PATH",  "data/day_2011.csv")

BASELINE_RMSE = float(os.getenv("BASELINE_RMSE", "315.7"))
BASELINE_MAE  = float(os.getenv("BASELINE_MAE",  "196.3"))
QUALITY_FACTOR = float(os.getenv("QUALITY_FACTOR", "1.00"))  
R2_MIN = float(os.getenv("R2_MIN", "0.80"))

RMSE_THRESHOLD = BASELINE_RMSE * QUALITY_FACTOR
MAE_THRESHOLD  = BASELINE_MAE  * QUALITY_FACTOR


# ======================
# LOAD MODEL + DATA
# ======================
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# If dteday exists, convert to month/day/weekday and drop it
if "dteday" in df.columns:
    df["dteday"] = pd.to_datetime(df["dteday"], dayfirst=True, errors="coerce")
    df["month"] = df["dteday"].dt.month
    df["day"] = df["dteday"].dt.day
    df["weekday"] = df["dteday"].dt.weekday
    df = df.drop(columns=["dteday"], errors="ignore")

# If mnth exists but month doesn't, rename it
if "mnth" in df.columns and "month" not in df.columns:
    df = df.rename(columns={"mnth": "month"})

# Split X/y
y = df["cnt"]
X = df.drop(columns=["cnt"])

# Align columns to what the model was trained on (prevents "feature names mismatch")
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    X = X.reindex(columns=expected_cols, fill_value=0)

# ======================
# METRICS
# ======================
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))
mae = mean_absolute_error(y, preds)
r2 = r2_score(y, preds)

print("=== Model Performance ===")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)
print("=========================")

# ======================
# QUALITY GATE
# ======================
fail = False

if rmse > RMSE_THRESHOLD:
    print(f"[FAIL] RMSE {rmse:.3f} > {RMSE_THRESHOLD:.3f}")
    fail = True
else:
    print(f"[PASS] RMSE {rmse:.3f} <= {RMSE_THRESHOLD:.3f}")

if mae > MAE_THRESHOLD:
    print(f"[FAIL] MAE {mae:.3f} > {MAE_THRESHOLD:.3f}")
    fail = True
else:
    print(f"[PASS] MAE {mae:.3f} <= {MAE_THRESHOLD:.3f}")

if r2 < R2_MIN:
    print(f"[FAIL] R2 {r2:.3f} < {R2_MIN:.3f}")
    fail = True
else:
    print(f"[PASS] R2 {r2:.3f} >= {R2_MIN:.3f}")

if fail:
    print("\nQuality Gate: ❌ FAILED")
    sys.exit(1)
else:
    print("\nQuality Gate: ✅ PASSED")
    sys.exit(0)
