# Customer Churn Prediction Platform

Production-ready churn prediction stack with:
- FastAPI inference API
- Streamlit frontend dashboard
- Versioned model artifacts
- Automated tests and CI gates

## Architecture

- Backend API: app/api.py
- Frontend UI: app/app.py
- Inference pipeline: src/pipelines/inference_pipeline.py
- Model registry: src/models/registry.py
- Model artifacts: models/v1/

## Quick Start (Local)

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn
```

3. Set environment variables.

Windows PowerShell:

```powershell
$env:MODEL_DIR="models"
$env:API_URL="http://localhost:8000"
$env:ALLOWED_ORIGINS="http://localhost:8501"
```

Linux/macOS:

```bash
export MODEL_DIR=models
export API_URL=http://localhost:8000
export ALLOWED_ORIGINS=http://localhost:8501
```

4. Start the API.

```bash
python -m uvicorn app.api:app --reload --port 8000
```

5. Start the dashboard.

```bash
streamlit run app/app.py
```

6. Open:
- Streamlit: http://localhost:8501
- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health

## API Endpoints

- GET /health
- GET /model-info
- POST /predict
- POST /predict-batch
- POST /monitor/report

## Model Artifacts

Current expected artifact layout:

```text
models/
  v1/
    model/
      model.joblib
      metadata.json
    scaler/
      scaler.joblib
      metadata.json
```

Use the same MODEL_DIR in every environment to avoid path drift.

## Testing and Quality Gates

Run locally:

```bash
ruff check . --select E9,F63,F7,F82
pytest -q
```

CI runs lint plus tests on every push and pull request.

## Docker

Build image:

```bash
docker build -t churn-api:latest .
```

Run API container:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models \
  -e ALLOWED_ORIGINS=http://localhost:8501 \
  -v $(pwd)/models:/models \
  churn-api:latest
```

## Public Launch Checklist

- Set ALLOWED_ORIGINS to your real frontend domain(s)
- Keep wildcard CORS disabled in production
- Store secrets in your deployment platform, not in code
- Pin dependency versions in requirements files
- Run CI gates before each release
- Tag releases and publish container images via workflow

## Release Notes (Current Baseline)

- Contract alignment fixed between frontend, API schema, and tests
- Test suite green: 21 passed
- Environment-based config overrides available via CFG__KEY__PATH
- MODEL_DIR and CORS origin controls added for deployment safety
