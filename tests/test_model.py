import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# -----------------------------
# Config (override via env in GitHub Actions)
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_rf_model.joblib")
DATA_PATH  = os.getenv("DATA_PATH", "data/day_2011.csv")
TARGET_COL = os.getenv("TARGET_COL", "cnt")

BASELINE_RMSE  = float(os.getenv("BASELINE_RMSE", "500"))
QUALITY_FACTOR = float(os.getenv("QUALITY_FACTOR", "0.95"))
THRESHOLD      = QUALITY_FACTOR * BASELINE_RMSE

print("=== Quality Gate Config ===")
print(f"MODEL_PATH      = {MODEL_PATH}")
print(f"DATA_PATH       = {DATA_PATH}")
print(f"BASELINE_RMSE   = {BASELINE_RMSE}")
print(f"QUALITY_FACTOR  = {QUALITY_FACTOR}")
print(f"THRESHOLD       = {THRESHOLD}")
print("===========================")

# -----------------------------
# Apply SAME cleaning as training
# -----------------------------
def align_features_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same transformations you used earlier:
    - parse dteday
    - extract month/day/weekday
    - drop dteday, mnth, old weekday
    - rename weekday_new -> weekday
    - drop year (as per your pipeline)
    Works even if df is already cleaned.
    """
    # If raw date exists, build derived features
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce", dayfirst=True)

        # Create derived columns 
        if "month" not in df.columns:
            df["month"] = df["dteday"].dt.month
        if "day" not in df.columns:
            df["day"] = df["dteday"].dt.day

        if "weekday" not in df.columns:
            df["weekday_new"] = df["dteday"].dt.weekday

    
    drop_cols = [c for c in ["dteday", "mnth"] if c in df.columns]
    
    if "weekday_new" in df.columns and "weekday" not in df.columns:
        df = df.rename(columns={"weekday_new": "weekday"})
    elif "weekday_new" in df.columns and "weekday" in df.columns:
        # keep weekday, drop weekday_new
        drop_cols.append("weekday_new")

    # Drop year 
    if "year" in df.columns:
        drop_cols.append("year")

    # Now actually drop selected columns
    drop_cols = list(dict.fromkeys(drop_cols))  # de-duplicate
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


# -----------------------------
# Load model + data
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[FAIL] Model file not found: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(DATA_PATH):
    print(f"[FAIL] Data file not found: {DATA_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    print(f"[FAIL] Target column '{TARGET_COL}' not found in dataset.")
    print("Columns:", list(df.columns))
    sys.exit(1)

# Apply same cleaning as training so feature columns match
df = align_features_like_training(df)

# Split
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------------
# Feature name alignment with model (prevents mismatch)
# -----------------------------
# scikit-learn stores training feature names in feature_names_in_ (for many models)
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]

    if missing:
        print("[FAIL] Missing required features:", missing)
        print("Current columns:", list(X.columns))
        sys.exit(1)

    # Reorder and drop extras to match exactly
    X = X[expected]
    if extra:
        print("[INFO] Dropping extra features not seen during training:", extra)

# -----------------------------
# Predict + evaluate
# -----------------------------
preds = model.predict(X)
rmse = float(np.sqrt(mean_squared_error(y, preds)))

print(f"RMSE = {rmse}")

# -----------------------------
# Quality Gate decision
# -----------------------------
if rmse <= THRESHOLD:
    print(f"[PASS] ✅ RMSE {rmse:.3f} <= {THRESHOLD:.3f}")
    sys.exit(0)
else:
    print(f"[FAIL] ❌ RMSE {rmse:.3f} > {THRESHOLD:.3f}")
    sys.exit(1)