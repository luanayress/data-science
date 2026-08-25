"""DEPRECATED: legacy v1 generator; use scripts/train_churn.py.

Regenerate models with proper serialization
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Create models directory
os.makedirs('models/v1', exist_ok=True)

print("📦 REGENERATING MODELS FOR DEPLOYMENT")
print("=" * 60)

# Load feature-engineered data
print("\n1️⃣  Loading feature-engineered data...")
df = pd.read_csv('data/processed/Customer_Churn_Final_Features.csv')
print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} features")

# Prepare features and target
print("\n2️⃣  Preparing features and target...")
features_to_use = ['NumOfProducts', 'Age_Squared_StandardScaled', 'Age_Tenure_Interaction_MinMaxScaled']
X = df[features_to_use]
y = df['Exited']
print(f"   ✓ Features: {features_to_use}")
print(f"   ✓ Churn distribution: {y.value_counts().to_dict()}")

# Split data
print("\n3️⃣  Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)} samples")
print(f"   ✓ Test: {len(X_test)} samples")

# Scale features
print("\n4️⃣  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ StandardScaler fitted and applied")

# Train Gradient Boosting model
print("\n5️⃣  Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)
gb_model.fit(X_train_scaled, y_train)
print(f"   ✓ Model trained")

# Evaluate
train_score = gb_model.score(X_train_scaled, y_train)
test_score = gb_model.score(X_test_scaled, y_test)
print(f"   ✓ Train accuracy: {train_score:.4f}")
print(f"   ✓ Test accuracy: {test_score:.4f}")

# Save models using joblib
print("\n6️⃣  Saving model artifacts with joblib...")

# Save model
model_path = 'models/v1/model.pkl'
joblib.dump(gb_model, model_path, compress=0)
print(f"   ✓ Model saved: {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

# Save scaler
scaler_path = 'models/v1/scaler.pkl'
joblib.dump(scaler, scaler_path, compress=0)
print(f"   ✓ Scaler saved: {scaler_path} ({os.path.getsize(scaler_path) / 1024:.1f} KB)")

print("\n" + "=" * 60)
print("✅ MODELS REGENERATED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Models v1 directory contents:")
for file in sorted(os.listdir('models/v1')):
    size = os.path.getsize(f'models/v1/{file}')
    if size > 1024*1024:
        print(f"   • {file}: {size / (1024*1024):.2f} MB")
    else:
        print(f"   • {file}: {size / 1024:.1f} KB")
