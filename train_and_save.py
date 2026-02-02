"""
Train models from scratch and save them for deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
num_features = ['NumOfProducts']
cat_features = []  # Add categorical features here if needed

# Feature engineering
df['Age_Squared'] = df['Age'] ** 2
df['Age_Tenure_Interaction'] = df['Age'] * df['Tenure']
num_features += ['Age_Squared', 'Age_Tenure_Interaction']

X = df[['NumOfProducts', 'Age_Squared', 'Age_Tenure_Interaction']]
y = df['Exited']
print(f"   ✓ Features: {X.columns.tolist()}")
print(f"   ✓ Churn distribution: {y.value_counts().to_dict()}")

# Split data
print("\n3️⃣  Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)} samples")
print(f"   ✓ Test: {len(X_test)} samples")


# Build preprocessing pipeline
print("\n4️⃣  Building preprocessing pipeline...")
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        # ("cat", cat_pipeline, cat_features)  # Add if categorical features exist
    ]
)

# Fit preprocessor on training data
X_train_transformed = preprocessor.fit_transform(X_train)
print(f"   ✓ Preprocessor fitted and applied")


# Train Gradient Boosting model
print("\n5️⃣  Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)
gb_model.fit(X_train_transformed, y_train)
print(f"   ✓ Model trained")


# Evaluate
train_score = gb_model.score(X_train_transformed, y_train)
X_test_transformed = preprocessor.transform(X_test)
test_score = gb_model.score(X_test_transformed, y_test)
print(f"   ✓ Train accuracy: {train_score:.4f}")
print(f"   ✓ Test accuracy: {test_score:.4f}")


# Save models
print("\n6️⃣  Saving model artifacts...")

# Save preprocessor
preprocessor_dir = 'models/v1/preprocessor'
os.makedirs(preprocessor_dir, exist_ok=True)
preprocessor_path = os.path.join(preprocessor_dir, 'preprocessor.joblib')
joblib.dump(preprocessor, preprocessor_path)
print(f"   ✓ Preprocessor saved: {preprocessor_path} ({os.path.getsize(preprocessor_path) / (1024*1024):.2f} MB)")

# Save model in MLOps layout
model_dir = 'models/v1/model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.joblib')
joblib.dump(gb_model, model_path)
print(f"   ✓ Model saved: {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

# Optionally remove old model.pkl if exists
old_model_pkl = 'models/v1/model.pkl'
if os.path.exists(old_model_pkl):
    os.remove(old_model_pkl)
    print(f"   ✓ Removed old artifact: {old_model_pkl}")

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
