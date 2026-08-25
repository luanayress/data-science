"""Migrate legacy .pkl model artifacts to MLOps layout with joblib and metadata.

- Converts:
  - models/gradient_boosting_model.pkl -> models/v1/model/model.joblib
  - models/scaler_standard.pkl -> models/v1/scaler/scaler.joblib
  - models/preprocessing_config.pkl -> models/v1/preprocessor/config.json
- Creates metadata.json files with mandatory fields.
"""

import os
import sys
import pickle
import joblib
import json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')
V1_DIR = os.path.join(MODELS_DIR, 'v1')

os.makedirs(V1_DIR, exist_ok=True)

summary = []

# Helper to write metadata
def write_metadata(path, meta):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# Try to infer features from model or fallback to data file
def infer_features(obj):
    # scikit-learn estimators often expose feature_names_in_
    if hasattr(obj, 'feature_names_in_'):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass
    # Try to read processed CSV as fallback
    proc_csv = os.path.join(ROOT, 'data', 'processed', 'Customer_Churn_Final_Features.csv')
    if os.path.exists(proc_csv):
        try:
            import pandas as pd
            df = pd.read_csv(proc_csv)
            # try dropping typical target columns
            for target in ['churn', 'Churn', 'target']:
                if target in df.columns:
                    df = df.drop(columns=[target])
                    break
            return list(df.columns)
        except Exception:
            pass
    return []

# Convert model (updated to use models/v1/model.pkl)
model_src = os.path.join(V1_DIR, 'model.pkl')
if os.path.exists(model_src):
    try:
        with open(model_src, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {model_src}: {e}")
        model = None

    if model is not None:
        model_dir = os.path.join(V1_DIR, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_dst = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_dst)

        metadata = {
            "component": "model",
            "model_type": type(model).__name__,
            "features": infer_features(model),
            "train_score": None,
            "saved_at": datetime.utcnow().isoformat() + 'Z'
        }
        write_metadata(os.path.join(model_dir, 'metadata.json'), metadata)
        summary.append(('model', model_dst))

# Convert scaler
scaler_src = os.path.join(MODELS_DIR, 'scaler_standard.pkl')
if os.path.exists(scaler_src):
    try:
        with open(scaler_src, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {scaler_src}: {e}")
        scaler = None

    if scaler is not None:
        scaler_dir = os.path.join(V1_DIR, 'scaler')
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_dst = os.path.join(scaler_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_dst)

        metadata = {
            "component": "scaler",
            "model_type": type(scaler).__name__,
            "features": infer_features(scaler),
            "train_score": None,
            "saved_at": datetime.utcnow().isoformat() + 'Z'
        }
        write_metadata(os.path.join(scaler_dir, 'metadata.json'), metadata)
        summary.append(('scaler', scaler_dst))

# Convert preprocessing config (dict) to JSON
prep_src = os.path.join(MODELS_DIR, 'preprocessing_config.pkl')
if os.path.exists(prep_src):
    try:
        with open(prep_src, 'rb') as f:
            prep = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {prep_src}: {e}")
        prep = None

    if prep is not None:
        prep_dir = os.path.join(V1_DIR, 'preprocessor')
        os.makedirs(prep_dir, exist_ok=True)
        prep_dst = os.path.join(prep_dir, 'config.json')
        try:
            with open(prep_dst, 'w', encoding='utf-8') as f:
                json.dump(prep, f, indent=2, ensure_ascii=False)
            metadata = {
                "component": "preprocessor",
                "type": "config",
                "saved_at": datetime.utcnow().isoformat() + 'Z'
            }
            write_metadata(os.path.join(prep_dir, 'metadata.json'), metadata)
            summary.append(('preprocessor', prep_dst))
        except Exception as e:
            print(f"Failed to write preprocessor config: {e}")

# Create README for v1
readme_path = os.path.join(V1_DIR, 'README.md')
if not os.path.exists(readme_path):
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('# Model version v1\n\n')
        f.write('Layout:\n')
        f.write('- model/model.joblib + metadata.json\n')
        f.write('- scaler/scaler.joblib + metadata.json\n')
        f.write('- preprocessor/config.json + metadata.json\n')
    summary.append(('readme', readme_path))

print('Migration summary:')
for item, path in summary:
    print(f' - {item}: {path}')

if not summary:
    print('No legacy artifacts found to migrate.')
else:
    print('Migration completed successfully.')


if __name__ == '__main__':
    pass
