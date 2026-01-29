import joblib

config = joblib.load('models/v1/preprocessing_config.pkl')
print('=== MODEL ARTIFACTS ===')
print(f'Churn Rate: {config["churn_rate"]:.2%}')
print(f'Features Used: {config["features"]}')
print('Models saved successfully in models/v1/')
