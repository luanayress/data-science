## App Architecture: FastAPI + Streamlit Split

**Status:** ✅ COMPLETED

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STREAMLIT FRONTEND (Visualization & UI)                   │
│  • Single Customer Prediction Form                          │
│  • Batch Predictions (CSV upload)                           │
│  • Model Information Dashboard                              │
│  • Analytics & Demo Data                                    │
│  • HTTP Requests to FastAPI                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                    HTTP Requests
                    (Port 8000)
                         │
┌────────────────────────▼────────────────────────────────────┐
│  FASTAPI BACKEND (Model Inference Server)                  │
│  • Model Loading & Caching                                 │
│  • Single Prediction Endpoint (/predict)                   │
│  • Batch Prediction Endpoint (/predict-batch)              │
│  • Model Info Endpoint (/model-info)                       │
│  • Health Check Endpoint (/health)                         │
│  • CORS Enabled for Frontend                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                      Python
                    Inference
                      Modules
                         │
┌────────────────────────▼────────────────────────────────────┐
│  ML INFERENCE PIPELINE                                     │
│  ├─ src/pipelines/inference_pipeline.py                    │
│  ├─ src/models/registry.py (Model Loading)                │
│  ├─ src/features/build_features.py                        │
│  └─ src/utils/paths.py, config.py                         │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
app/
├── __init__.py
├── api.py              # FastAPI backend (NEW)
├── app.py              # Streamlit frontend (`app/app.py`)
├── schema.py           # Pydantic schemas
└── model_loader.py     # Optional: Direct model loading
```

### FastAPI Backend (api.py)

**Purpose:** RESTful API server for model inference

**Key Features:**
- ✅ Model loading on startup
- ✅ CORS middleware for Streamlit cross-origin requests
- ✅ Health check endpoint
- ✅ Single prediction endpoint with confidence
- ✅ Batch prediction endpoint
- ✅ Model information endpoint
- ✅ Automatic API documentation (Swagger UI)

**Endpoints:**

| Endpoint | Method | Purpose | Input |
|----------|--------|---------|-------|
| `/` | GET | Root info | - |
| `/health` | GET | Health check | - |
| `/model-info` | GET | Model metadata | - |
| `/predict` | POST | Single prediction | `PredictionRequest` |
| `/predict-batch` | POST | Batch predictions | `BatchPredictionRequest` |

**Usage:**
```bash
# Start API server
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Or directly
python app/api.py
```

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Streamlit Frontend (`app/app.py`)

**Purpose:** Interactive visualization and prediction interface

**Key Features:**
- ✅ 4-page navigation system
- ✅ Single customer prediction form
- ✅ Batch predictions with CSV upload
- ✅ Model information display
- ✅ Analytics dashboard with sample data
- ✅ API connectivity with health checks
- ✅ Interactive Plotly visualizations
- ✅ CSV export of predictions
- ✅ Gauge charts for probability visualization

**Pages:**

1. **Single Prediction**
   - Customer input form
   - Real-time churn prediction
   - Probability gauge chart
   - Risk recommendations

2. **Batch Predictions**
   - CSV file upload
   - Bulk prediction processing
   - Results preview and statistics
   - CSV download

3. **Model Info**
   - Model metadata display
   - Feature names and count
   - Training metrics

4. **Analytics**
   - Sample data analysis
   - Churn distribution pie chart
   - Probability distribution
   - Tenure vs charges scatter plot
   - Confidence distribution box plot

**Usage:**
```bash
# Start Streamlit frontend
streamlit run app/app.py

# Custom configuration
streamlit run app/app.py --logger.level=info
```

Access at: `http://localhost:8501`

### Schemas (schema.py)

Pydantic models for validation:
- `PredictionRequest` - Single prediction input
- `PredictionResponse` - Single prediction output
- `BatchPredictionRequest` - Multiple predictions input
- `BatchPredictionResponse` - Multiple predictions output
- `ModelInfo` - Model metadata
- `HealthCheck` - API health status

### Data Flow

```
User Input (Streamlit)
    ↓
HTTP POST Request
    ↓
FastAPI Endpoint Receiver
    ↓
Pydantic Schema Validation
    ↓
DataFrame Conversion
    ↓
Feature Preprocessing (InferencePipeline)
    ↓
Model Prediction (GradientBoosting)
    ↓
Confidence Scoring
    ↓
JSON Response (PredictionResponse)
    ↓
HTTP Response to Streamlit
    ↓
Visualization & Display
```

### Benefits of This Architecture

#### **Separation of Concerns**
✅ Backend (FastAPI) - Pure inference logic
✅ Frontend (Streamlit) - Pure visualization logic
✅ Schema (Pydantic) - Data validation

#### **Scalability**
✅ API can be deployed independently
✅ Frontend can be scaled separately
✅ Multiple instances can share backend
✅ Easy to add other clients (mobile, web, CLI)

#### **Reliability**
✅ API health checks
✅ Error handling and logging
✅ Timeout management
✅ CORS configuration

#### **Development**
✅ Test API independently
✅ Mock API for frontend development
✅ Swagger documentation auto-generated
✅ Type safety with Pydantic

#### **Deployment**
✅ API: Docker + Gunicorn/Uvicorn
✅ Frontend: Streamlit Cloud / Docker
✅ Easy CI/CD pipeline setup
✅ Independent versioning

### Running the Full Application

**Terminal 1: Start API Server**
```bash
cd "C:\Users\Luan\Desktop\Data Science"
python -m uvicorn app.api:app --reload --port 8000
```

**Terminal 2: Start Streamlit Frontend**
```bash
cd "C:\Users\Luan\Desktop\Data Science"
streamlit run app/app.py
```

**Access Points:**
- Streamlit UI: `http://localhost:8501`
- API Docs: `http://localhost:8000/docs`
- API Health: `http://localhost:8000/health`

### Environment Variables

**Optional - Create `.streamlit/secrets.toml`:**
```toml
API_URL = "http://localhost:8000"
```

### Dependencies

**New Package Required:**
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `requests` - HTTP client (Streamlit)
- `plotly` - Interactive visualizations

**Install:**
```bash
pip install fastapi uvicorn plotly requests
```

### Configuration Files

- `configs/training.yaml` - Training parameters
- `configs/inference.yaml` - Inference settings
- `configs/features.yaml` - Feature definitions

### Next Steps

1. **Train and Save Model**
   ```python
   from src.pipelines.training_pipeline import run_training_pipeline
   results = run_training_pipeline(save_model=True)
   ```

2. **Start API Server**
   ```bash
   python -m uvicorn app.api:app --reload
   ```

3. **Launch Streamlit Frontend**
   ```bash
   streamlit run app/app.py
   ```

4. **Deploy (Optional)**
   - API: Deploy to cloud (AWS, GCP, Azure)
   - Frontend: Deploy to Streamlit Cloud

### Testing the API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Using Python:**
```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "SeniorCitizen": 0,
    "Tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1570.0,
    # ... other fields
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

### Monitoring

**API Logs:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Streamlit Logs:**
```
2026-01-28 10:00:00 - Making prediction...
2026-01-28 10:00:01 - Prediction complete
```

### Troubleshooting

**API not connecting:**
- Check if API is running: `curl http://localhost:8000/health`
- Check port 8000 is not in use
- Check firewall settings

**Model not loading:**
- Verify `models/v1/model.pkl` exists
- Check `models/v1/metadata.json` is valid
- Check file permissions

**Streamlit errors:**
- Check API URL in sidebar
- Verify network connectivity
- Check browser console for JavaScript errors

---

**Status:** ✅ Architecture updated successfully. Ready for training and deployment!
