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
export MODEL_DIR=artifacts/models
export DATA_DIR=data
export LOG_DIR=logs
export API_URL=http://localhost:8000
```

On Windows (PowerShell):

```powershell
$env:MODEL_DIR="artifacts/models"
$env:DATA_DIR="data"
$env:LOG_DIR="logs"
$env:API_URL="http://localhost:8000"
```

---

## 3. Training Pipeline (Batch)

**What this does**:

* Builds features
* Validates feature contract
* Fits preprocessing + GradientBoostingClassifier
* Saves a single pipeline artifact

```bash
python train_and_save.py
```

Artifacts produced:

* `MODEL_DIR/model_pipeline.joblib`

---

## 4. Test Suite (Mandatory)

```bash
pytest --maxfail=3 --disable-warnings -v
```

If tests fail, **do not continue**.

---

## 5. Start API Server (Inference)

Must be running **before** Streamlit.

```bash
python -m uvicorn app.api:app --reload --port 8000
```

Health check:

```bash
http://localhost:8000/health
```

---

## 6. Run Streamlit Dashboard

```bash
streamlit run app/app.py
```

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
  -e MODEL_DIR=/artifacts/models \
  -v $(pwd)/artifacts:/artifacts \
  ml-project
```

### 7.3 Run API in Docker

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/artifacts/models \
  -v $(pwd)/artifacts:/artifacts \
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
