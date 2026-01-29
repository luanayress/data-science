# Quick Start Guide: FastAPI + Streamlit Architecture

## Overview

Your project now uses a **two-tier architecture**:

```
Streamlit (Frontend)  ←→  FastAPI (Backend)  ←→  ML Model
```

---

## Getting Started

### 1. **Set Virtual Environment**
```powershell
cd "C:\Users\Luan\Desktop\Data Science"
.\.venv\Scripts\Activate.ps1
```

### 2. **Set PYTHONPATH**
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### 3. **Train Model** (One Time)
```powershell
.\.venv\Scripts\python.exe -c "
from src.pipelines.training_pipeline import run_training_pipeline
results = run_training_pipeline(save_model=True, version='v1')
print('✅ Model trained and saved!')
"
```

---

## Running the Application

### **Option A: Both in Same Terminal (Sequential)**

```powershell
# Terminal: Start API first
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --port 8000

# Wait for "Application startup complete", then...
# Open new terminal in same folder and run:
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

### **Option B: Separate Terminals (Recommended)**

**Terminal 1 (API):**
```powershell
cd "C:\Users\Luan\Desktop\Data Science"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```powershell
cd "C:\Users\Luan\Desktop\Data Science"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

---

## Access Points

| Component | URL | Purpose |
|-----------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Predict & Analyze |
| **API Docs** | http://localhost:8000/docs | Swagger Documentation |
| **API Health** | http://localhost:8000/health | System Status |
| **ReDoc** | http://localhost:8000/redoc | API Reference |

---

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Single Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "SeniorCitizen": 0,
  "Tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1570.0,
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
```

### Batch Predictions
```bash
POST http://localhost:8000/predict-batch
Content-Type: application/json

{
  "data": [
    { customer1 },
    { customer2 },
    ...
  ]
}
```

---

## File Structure

```
app/
├── api.py                # FastAPI Backend
├── app.py                # Streamlit Frontend
├── schema.py             # Pydantic Schemas
└── model_loader.py       # Optional Direct Loading

src/
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── models/
│   ├── registry.py
│   └── ...
└── ...

models/v1/
├── model.pkl             # Trained model
├── scaler.pkl            # Preprocessor
└── metadata.json         # Model info

configs/
├── training.yaml         # Training config
├── inference.yaml        # Inference config
└── features.yaml         # Feature config
```

---

## Key Commands Reference

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Set Python path
$env:PYTHONPATH = (Get-Location).Path

# Start API
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload

# Start Streamlit
.\.venv\Scripts\python.exe -m streamlit run app/app.py

# Test API health
curl http://localhost:8000/health

# Run tests
.\.venv\Scripts\python.exe tests/test_features.py
.\.venv\Scripts\python.exe tests/test_model.py
.\.venv\Scripts\python.exe tests/test_api.py

# Train model
.\.venv\Scripts\python.exe -c "from src.pipelines.training_pipeline import run_training_pipeline; run_training_pipeline(save_model=True)"

# Check Python version
.\.venv\Scripts\python.exe --version

# List installed packages
.\.venv\Scripts\pip.exe list
```

---

## Troubleshooting

### API won't start
```
1. Check port 8000 is free: netstat -ano | findstr :8000
2. Kill process if needed: taskkill /PID <PID> /F
3. Try different port: --port 8001
```

### Model not found
```
1. Verify: models/v1/model.pkl exists
2. Train model: python train_and_save.py
3. Check path: src/utils/paths.py
```

### Streamlit can't reach API
```
1. Verify API is running on port 8000
2. Check firewall settings
3. Test: curl http://localhost:8000/health
```

### Import errors
```
1. Activate venv: .\.venv\Scripts\Activate.ps1
2. Set PYTHONPATH: $env:PYTHONPATH = (Get-Location).Path
3. Verify PYTHONPATH: echo $env:PYTHONPATH
```

---

## Features by Page

### 📋 **Single Prediction**
- Customer data input form
- Real-time churn prediction
- Probability gauge visualization
- Risk level indicator
- Actionable recommendations

### 📁 **Batch Predictions**
- CSV file upload
- Process multiple customers
- Results statistics
- CSV download of predictions
- Churn distribution summary

### ℹ️ **Model Info**
- Model type and version
- Training metrics
- Feature list and count
- Model performance details

### 📊 **Analytics**
- Sample data dashboard
- Churn distribution pie chart
- Probability distribution histogram
- Tenure vs charges scatter plot
- Confidence level analysis

---

## Development Tips

### Test API before Frontend
```powershell
# Use curl or Python
.\.venv\Scripts\python.exe -c "
import requests
r = requests.get('http://localhost:8000/health')
print(r.json())
"
```

### Debug Mode
```powershell
# API with verbose logging
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --log-level debug

# Streamlit with logger level
.\.venv\Scripts\python.exe -m streamlit run app/app.py --logger.level=debug
```

### Add Custom Endpoint
Edit `app/api.py` and add new route:
```python
@app.post("/custom-endpoint")
async def custom_endpoint(request: CustomRequest):
    # Your logic here
    return response
```

---

## Performance Notes

- **Single Prediction:** ~100-200ms
- **Batch (100 samples):** ~1-2 seconds
- **API Startup:** ~2-3 seconds
- **Model Load:** ~1-2 seconds

---

## Next Steps

1. ✅ Train model and verify it saves to `models/v1/`
2. ✅ Start API server and test endpoints with Swagger UI
3. ✅ Launch Streamlit and test form predictions
4. ✅ Upload CSV for batch predictions
5. ✅ Review analytics dashboard with sample data
6. ✅ Deploy to cloud (AWS, GCP, Heroku, etc.)

---

## Documentation

- **Architecture:** See `APP_ARCHITECTURE.md`
- **Modules:** See `PHASE_3_COMPLETE.md`
- **Project Structure:** See `QUICK_REFERENCE.txt`
- **API Docs:** Run API and visit `http://localhost:8000/docs`

---

**Ready to go!** 🚀
