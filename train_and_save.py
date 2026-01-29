"""
Train models from scratch and save them for deployment
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Create models directory
os.makedirs('models/v1', exist_ok=True)

print("📦 TRAINING & SAVING MODELS FOR DEPLOYMENT")
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
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Fit on training data
X_train_scaled = scaler_standard.fit_transform(X_train)
X_test_scaled = scaler_standard.transform(X_test)
print(f"   ✓ StandardScaler fitted and applied")

# MinMaxScaler (just for consistency with original pipeline)
scaler_minmax.fit(X_train)
print(f"   ✓ MinMaxScaler fitted")

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

# Create preprocessing config
preprocessing_config = {
    'features': features_to_use,
    'churn_rate': float(y.mean()),
    'feature_mean': X_train.mean().to_dict(),
    'feature_std': X_train.std().to_dict(),
}

# Save models
print("\n6️⃣  Saving model artifacts...")

# Save model
model_path = 'models/v1/model.pkl'
joblib.dump(gb_model, model_path)
print(f"   ✓ Model saved: {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

# Save scalers
scaler_std_path = 'models/v1/scaler_standard.pkl'
joblib.dump(scaler_standard, scaler_std_path)
print(f"   ✓ StandardScaler saved: {scaler_std_path} ({os.path.getsize(scaler_std_path) / 1024:.1f} KB)")

scaler_minmax_path = 'models/v1/scaler_minmax.pkl'
joblib.dump(scaler_minmax, scaler_minmax_path)
print(f"   ✓ MinMaxScaler saved: {scaler_minmax_path} ({os.path.getsize(scaler_minmax_path) / 1024:.1f} KB)")

# Save config
config_path = 'models/v1/preprocessing_config.pkl'
joblib.dump(preprocessing_config, config_path)
print(f"   ✓ Config saved: {config_path} ({os.path.getsize(config_path) / 1024:.1f} KB)")

print("\n" + "=" * 60)
print("✅ ALL MODELS SAVED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Models v1 directory contents:")
for file in sorted(os.listdir('models/v1')):
    size = os.path.getsize(f'models/v1/{file}')
    if size > 1024*1024:
        print(f"   • {file}: {size / (1024*1024):.2f} MB")
    else:
        print(f"   • {file}: {size / 1024:.1f} KB")

print("\n🚀 Ready to deploy! Run: streamlit run app.py")
