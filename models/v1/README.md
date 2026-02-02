# Model version v1

Layout:
- model/model.joblib + metadata.json
- scaler/scaler.joblib + metadata.json
- preprocessor/config.json + metadata.json

Minimum `metadata.json` for `model` (example):

```json
{
  "component": "model",
  "model_type": "GradientBoostingClassifier",
  "features": ["age", "tenure", "..."] ,
  "train_score": 0.87,
  "saved_at": "2026-01-28T08:30:00Z"
}
```
