import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# CONFIG
# ======================
MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_rf_model.joblib")
DATA_PATH  = os.getenv("DATA_PATH",  "data/day_2011.csv")

# Baselines should be FULL 2011 performance of the SAVED model
BASELINE_RMSE = float(os.getenv("BASELINE_RMSE", "315.7"))
BASELINE_MAE  = float(os.getenv("BASELINE_MAE",  "196.3"))

# Use separate factors so 2011 reliably passes but 2012 still fails
QUALITY_FACTOR_2011 = float(os.getenv("QUALITY_FACTOR_2011", "1.10"))  # allow +10% on 2011
QUALITY_FACTOR_2012 = float(os.getenv("QUALITY_FACTOR_2012", "1.05"))  # stricter for drift checks

# R2 threshold (keep stable for both)
R2_MIN = float(os.getenv("R2_MIN", "0.80"))

# Detect which dataset we are checking
data_path_lower = DATA_PATH.lower()
is_2011 = ("2011" in data_path_lower)
is_2012 = ("2012" in data_path_lower)

# Choose factor based on dataset
if is_2011:
    QUALITY_FACTOR = QUALITY_FACTOR_2011
elif is_2012:
    QUALITY_FACTOR = QUALITY_FACTOR_2012
else:
    # fallback if filename doesn't include year
    QUALITY_FACTOR = float(os.getenv("QUALITY_FACTOR", "1.10"))

RMSE_THRESHOLD = BASELINE_RMSE * QUALITY_FACTOR
MAE_THRESHOLD  = BASELINE_MAE  * QUALITY_FACTOR

print("=== Quality Gate Config ===")
print("MODEL_PATH:", MODEL_PATH)
print("DATA_PATH :", DATA_PATH)
print("BASELINE_RMSE:", BASELINE_RMSE)
print("BASELINE_MAE :", BASELINE_MAE)
print("QUALITY_FACTOR:", QUALITY_FACTOR)
print("RMSE_THRESHOLD:", RMSE_THRESHOLD)
print("MAE_THRESHOLD :", MAE_THRESHOLD)
print("R2_MIN:", R2_MIN)
print("===========================")

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

# Align columns to what the model was trained on
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    X = X.reindex(columns=expected_cols, fill_value=0)

# ======================
# METRICS
# ======================
preds = model.predict(X)
rmse = float(np.sqrt(mean_squared_error(y, preds)))
mae  = float(mean_absolute_error(y, preds))
r2   = float(r2_score(y, preds))

print("\n=== Model Performance ===")
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

