"""
Reorganize models to new directory structure:
models/v1/
├── model/
│   ├── model.joblib
│   └── metadata.json
├── scaler/
│   ├── scaler.joblib
│   └── metadata.json
"""
import shutil
import joblib
import json
from pathlib import Path

models_v1_dir = Path("models/v1")
print("📁 REORGANIZING MODEL STRUCTURE")
print("=" * 60)

# Backup existing structure
backup_dir = Path("models/v1_backup")
if models_v1_dir.exists():
    print(f"\n1️⃣  Backing up existing models to {backup_dir}...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(models_v1_dir, backup_dir)
    print(f"   ✓ Backup created")
    
    # Clean up v1 directory
    for file in models_v1_dir.glob("*"):
        if file.is_file():
            file.unlink()

# Create new structure
print("\n2️⃣  Creating new directory structure...")
model_dir = models_v1_dir / "model"
scaler_dir = models_v1_dir / "scaler"
model_dir.mkdir(parents=True, exist_ok=True)
scaler_dir.mkdir(parents=True, exist_ok=True)
print(f"   ✓ Created model/ and scaler/ directories")

# Load old artifacts from backup
print("\n3️⃣  Moving artifacts to new structure...")
backup_model = backup_dir / "model.pkl"
backup_scaler = backup_dir / "scaler.pkl"

if backup_model.exists():
    model_obj = joblib.load(backup_model)
    joblib.dump(model_obj, model_dir / "model.joblib", compress=0)
    print(f"   ✓ Moved model to model/model.joblib")

if backup_scaler.exists():
    scaler_obj = joblib.load(backup_scaler)
    joblib.dump(scaler_obj, scaler_dir / "scaler.joblib", compress=0)
    print(f"   ✓ Moved scaler to scaler/scaler.joblib")

# Create metadata files
print("\n4️⃣  Creating metadata files...")

model_metadata = {
    "component": "model",
    "type": "GradientBoostingClassifier",
    "file": "model.joblib",
    "features": [
        "NumOfProducts",
        "Age_Squared_StandardScaled",
        "Age_Tenure_Interaction_MinMaxScaled"
    ],
    "train_accuracy": 0.8539,
    "test_accuracy": 0.8325,
    "created_at": "2026-01-28"
}

scaler_metadata = {
    "component": "scaler",
    "type": "StandardScaler",
    "file": "scaler.joblib",
    "created_at": "2026-01-28"
}

with open(model_dir / "metadata.json", 'w') as f:
    json.dump(model_metadata, f, indent=2)
print(f"   ✓ Created model/metadata.json")

with open(scaler_dir / "metadata.json", 'w') as f:
    json.dump(scaler_metadata, f, indent=2)
print(f"   ✓ Created scaler/metadata.json")

# Display structure
print("\n" + "=" * 60)
print("✅ REORGANIZATION COMPLETE!")
print("=" * 60)
print("\n📁 New structure:")
print("models/v1/")
for component_dir in sorted(models_v1_dir.iterdir()):
    if component_dir.is_dir():
        print(f"├── {component_dir.name}/")
        for file in sorted(component_dir.iterdir()):
            size = file.stat().st_size
            if size > 1024:
                print(f"│   ├── {file.name} ({size / 1024:.1f} KB)")
            else:
                print(f"│   ├── {file.name} ({size} bytes)")

print(f"\n💾 Backup available at: {backup_dir}")
