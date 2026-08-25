# Project Runbook — End‑to‑End Execution

This document defines the **single, authoritative way** to run this project locally or inside Docker. Copilot, CI, and humans should follow these steps **in order**. Skipping steps is unsupported.

---

## 0. Guarantees

If these steps are followed:

* Training, inference, API, and dashboard **will work**
* Feature contracts are enforced
* No silent preprocessing or model drift occurs

If something fails, the failure is **intentional and diagnostic**.

---

## 1. Environment Setup (Local)


### 1.1 Activate existing virtual environment (recommended)

If `.venv` already exists, activate it:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
# If you see a security error, run:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# Then retry activation
```

**Windows (CMD)**

```cmd
.venv\Scripts\activate.bat
```

**Linux / Mac**

```bash
source .venv/bin/activate
```

If you need to create a new environment:

```bash
python -m venv .venv
```

---


### 1.2 Install dependencies (deterministic)

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Ensure FastAPI and Uvicorn are installed for API server
pip install fastapi uvicorn
```

---

## 2. Required Environment Variables

These defaults are safe for local and Docker execution.

```bash
export MODEL_DIR=models
export DATA_DIR=data
export LOG_DIR=logs
export API_URL=http://localhost:8000
export ALLOWED_ORIGINS=http://localhost:8501
```

On Windows (PowerShell):

```powershell
$env:MODEL_DIR="models"
$env:MODEL_VERSION="v2"
$env:DATA_DIR="data"
$env:LOG_DIR="logs"
$env:API_URL="http://localhost:8000"
$env:ALLOWED_ORIGINS="http://localhost:8501"
```

---

## 3. Training Pipeline (Batch)

**What this does**:

* Builds features
* Validates feature contract
* Fits preprocessing + GradientBoostingClassifier
* Saves a single pipeline artifact

```bash
python scripts/train_churn.py
```

Artifacts produced:

* `MODEL_DIR/v2/pipeline.joblib`
* `MODEL_DIR/v2/metadata.json`

---

## 4. Test Suite (Mandatory)

```bash
pytest --maxfail=3 --disable-warnings -v
```

If tests fail, **do not continue**.

### 4.1 Champion/challenger experiment (optional)

```bash
python scripts/compare_churn_models.py
```

This keeps the v2 holdout split, performs model selection and threshold analysis only on training/CV data, and writes reports to `reports/model-comparison/`. A qualifying challenger is stored under `models/v3/`, but the API default remains v2.

### 4.2 Promotion evaluation and shadow validation

```bash
python scripts/evaluate_model_promotion.py
```

Business assumptions live in `configs/churn_business.yaml`. To validate shadow inference without changing client responses:

```powershell
$env:MODEL_VERSION="v2"
$env:SHADOW_MODEL_VERSION="v3"
python -m uvicorn app.api:app --reload --port 8000
```

Rollback is explicit: set `MODEL_VERSION=v2`, clear the shadow/profile variables, and restart the API. See `docs/model-promotion.md`.

---

## 5. Start API Server (Inference)

Must be running **before** Streamlit.

```bash
python -m uvicorn app.api:app --reload --port 8000
```

The lifespan loads `models/v2/pipeline.joblib` once at startup. `MODEL_VERSION` overrides `configs/inference.yaml`; the YAML value overrides the built-in `v2` default.

Health check:

```bash
http://localhost:8000/health
```

---

## 6. Run Streamlit Dashboard

```bash
streamlit run app/app.py
```

No `.streamlit/secrets.toml` file is required for local execution. `API_URL` falls back to the URL configured in `configs/inference.yaml`.

Dashboard URL:

```text
http://localhost:8501
```

---

## 7. Docker (Optional but Supported)

### 7.1 Build image

```bash
docker build -t ml-project .
```

### 7.2 Run training in Docker

```bash
docker run --rm \
  -e MODEL_DIR=/models \
  -v $(pwd)/models:/models \
  ml-project
```

### 7.3 Run API in Docker

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models \
  -e ALLOWED_ORIGINS=http://localhost:8501 \
  -v $(pwd)/models:/models \
  ml-project \
  python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## 8. What Will Fail (By Design)

| Scenario             | Result                        |
| -------------------- | ----------------------------- |
| Missing features     | Hard failure with diagnostics |
| Unexpected columns   | Hard failure                  |
| Wrong types          | Hard failure                  |
| Manual preprocessing | Impossible                    |
| API without model    | Hard failure                  |

### 8.1 Expanded bank challenger (v4)

Train and reproduce the feature ablation with `python scripts/train_feature_expansion.py`. The generated reports
are stored in `reports/feature-expansion/`; the complete artifact is stored in `models/v4/`.

Use `MODEL_VERSION=v2` and `SHADOW_MODEL_VERSION=v4` for observation. Do not set `MODEL_VERSION=v4` in production
until business cost, capacity and latency constraints are approved. Roll back by restoring `MODEL_VERSION=v2`.

Current readiness is `NOT_READY_FOR_CANARY`. Read `reports/feature-expansion/readiness_decision.json` before any
rollout. Canary must remain disabled while any gate is `FAIL` or `BLOCKED`; currently fairness, financial value,
campaign capacity and latency SLO block progression. Shadow comparisons require complete v4 bank payloads.

---

## 9. Rules for Copilot / CI

* ❌ Do NOT skip steps
* ❌ Do NOT add preprocessing outside pipeline
* ❌ Do NOT modify feature names
* ✅ Always run tests before Streamlit
* ✅ API must start before dashboard

---

## 10. Source of Truth

This file overrides:

* README snippets
* Personal workflows
* IDE shortcuts

If this file changes, CI and onboarding **must** change with it.
