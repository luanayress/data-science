# Customer Churn Prediction Platform

Production-ready churn prediction stack with:
- FastAPI inference API
- Streamlit frontend dashboard
- Versioned model artifacts
- Automated tests and CI gates

## Architecture

- Backend bootstrap/lifespan: app/api.py
- FastAPI routers: app/routers/
- Application services: app/services/
- Dependency injection: app/dependencies.py
- Frontend UI: app/app.py
- Frontend HTTP client and analytics provider: app/frontend/services/
- Inference pipeline: src/pipelines/inference_pipeline.py
- Model registry: src/models/registry.py
- Model artifacts: models/v2/ (`models/v1/` is retained for rollback)

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
```

3. Set environment variables.

Windows PowerShell:

```powershell
$env:MODEL_DIR="models"
$env:MODEL_VERSION="v2"
$env:API_URL="http://localhost:8000"
$env:ALLOWED_ORIGINS="http://localhost:8501"
```

Linux/macOS:

```bash
export MODEL_DIR=models
export MODEL_VERSION=v2
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

Configuration precedence is environment variable, `configs/inference.yaml`, then a safe default. Supported application variables include `MODEL_VERSION`, `MODEL_DIR`, `API_HOST`, `API_PORT`, `API_URL`, `ALLOWED_ORIGINS`, `PREDICTION_THRESHOLD`, `LOG_LEVEL`, `APP_ENV`, `API_TIMEOUT`, and `API_BATCH_TIMEOUT`.

## Model Artifacts

Current active artifact layout:

```text
models/
  v2/
    pipeline.joblib
    metadata.json
```

Train v2 from raw data with:

```bash
python scripts/train_churn.py
```

Run the reproducible champion/challenger experiment with:

```bash
python scripts/compare_churn_models.py
```

Phase 2B produced a calibrated `models/v3/` challenger and reports under `reports/model-comparison/`. The application default remains `v2`; validate v3 explicitly with `MODEL_VERSION=v3` before promotion.

Evaluate promotion assumptions without retraining:

```bash
python scripts/evaluate_model_promotion.py
```

Optional shadow mode uses `MODEL_VERSION=v2` plus `SHADOW_MODEL_VERSION=v3`; v2 remains the HTTP response source. `CHURN_THRESHOLD_PROFILE` accepts `default`, `balanced`, `high_recall`, or `high_precision` when present in model metadata.

Feature expansion is available as the non-default `v4` challenger. It aligns the request/dashboard with the bank
dataset and selects ten leakage-safe features by training-only cross-validation:

```bash
python scripts/train_feature_expansion.py
```

See `docs/churn-model-v4.md`. Validate with `SHADOW_MODEL_VERSION=v4`; production remains on v2 until an explicit
promotion decision.

The legacy rollback layout remains under:

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
- Leakage-safe raw-data pipeline persisted as v2
- Test suite green: 68 passed
- Environment-based config overrides available via CFG__KEY__PATH
- MODEL_DIR and CORS origin controls added for deployment safety
- FastAPI routers/services, lifespan startup, and dependency injection added without changing HTTP contracts
- Streamlit HTTP calls centralized in ApiClient; analytics is explicitly labeled DEMO DATA
