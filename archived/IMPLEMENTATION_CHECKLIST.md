# Project Structure Migration Checklist

**Project:** Customer Churn Prediction  
**Status:** Documentation Updated ✅ | Implementation Pending  
**Last Updated:** January 28, 2026

---

## 📋 Directory Creation

### Create Directory Structure
- [ ] `app/` - Application layer (dashboard & API)
- [ ] `src/` - Core ML modules
  - [ ] `src/data/` - Data loading and validation
  - [ ] `src/features/` - Feature engineering
  - [ ] `src/models/` - Model training and evaluation
  - [ ] `src/pipelines/` - End-to-end workflows
  - [ ] `src/utils/` - Configuration and logging
- [ ] `notebooks/` - Jupyter notebooks (exploration only)
- [ ] `models/v1/` - Versioned model artifacts
- [ ] `data/` - Data management
  - [ ] `data/raw/` - Raw input data
  - [ ] `data/processed/` - Processed features
  - [ ] `data/external/` - External data sources
- [ ] `tests/` - Unit tests
- [ ] `configs/` - Configuration files
- [ ] `.github/workflows/` - CI/CD pipelines

---

## 🔄 File Migration

### Notebook Files
- [ ] Rename `exploratory_analysis.ipynb` → `notebooks/01_eda.ipynb`
- [ ] Rename `feature_eng.ipynb` → `notebooks/02_feature_engineering.ipynb`
- [ ] Rename `modeling.ipynb` → `notebooks/03_modeling.ipynb`

### Data Files
- [ ] Move `Customer-Churn-Records.csv` → `data/raw/Customer-Churn-Records.csv`
- [ ] Move `Customer_Churn_Final_Features.csv` → `data/processed/`
- [ ] Move `Customer_Churn_Engineered_Features.csv` → `data/processed/`

### Documentation Files
- [ ] Keep: `DEPLOYMENT_GUIDE.md` (already updated)
- [ ] Keep: `README.md` (already updated)
- [ ] Keep: `QUICK_REFERENCE.txt` (already updated)
- [ ] Keep: `DEPLOYMENT_INDEX.txt` (already updated)
- [ ] Keep: `DEPLOYMENT_SUMMARY.txt` (already updated)
- [ ] Keep: `Feature_Engineering_Report.txt`

---

## 📝 Code File Creation

### app/ Directory
- [ ] Create `app/__init__.py` - Package initialization
- [ ] Create `app/app.py` - Streamlit dashboard (~15 KB)
- [ ] Create `app/model_loader.py` - Model loading utilities (~8 KB)
- [ ] Create `app/schema.py` - Data validation schemas (~3 KB)

### src/data/ Directory
- [ ] Create `src/data/__init__.py` - Package initialization
- [ ] Create `src/data/load_data.py` - Data loading functions
- [ ] Create `src/data/validation.py` - Data validation
- [ ] Create `src/data/split.py` - Train/test splitting

### src/features/ Directory
- [ ] Create `src/features/__init__.py` - Package initialization
- [ ] Create `src/features/build_features.py` - Feature engineering pipeline
- [ ] Create `src/features/transformers.py` - Custom transformers

### src/models/ Directory
- [ ] Create `src/models/__init__.py` - Package initialization
- [ ] Create `src/models/train.py` - Model training logic
- [ ] Create `src/models/evaluate.py` - Model evaluation metrics
- [ ] Create `src/models/predict.py` - Prediction functions
- [ ] Create `src/models/registry.py` - Model registry/tracking

### src/pipelines/ Directory
- [ ] Create `src/pipelines/__init__.py` - Package initialization
- [ ] Create `src/pipelines/training_pipeline.py` - End-to-end training
- [ ] Create `src/pipelines/inference_pipeline.py` - End-to-end inference

### src/utils/ Directory
- [ ] Create `src/utils/__init__.py` - Package initialization
- [ ] Create `src/utils/config.py` - Configuration management
- [ ] Create `src/utils/logger.py` - Logging utilities
- [ ] Create `src/utils/paths.py` - Path utilities

### tests/ Directory
- [ ] Create `tests/__init__.py` - Package initialization
- [ ] Create `tests/test_features.py` - Feature engineering tests
- [ ] Create `tests/test_model.py` - Model tests
- [ ] Create `tests/test_api.py` - API endpoint tests

### configs/ Directory
- [ ] Create `configs/training.yaml` - Training configuration
- [ ] Create `configs/inference.yaml` - Inference configuration
- [ ] Create `configs/features.yaml` - Feature configuration

### .github/workflows/ Directory
- [ ] Create `.github/workflows/ci.yml` - Continuous integration
- [ ] Create `.github/workflows/deploy.yml` - Deployment pipeline

---

## 🛠️ Build & Configuration Files

- [ ] Create `Dockerfile` - Container configuration
- [ ] Create `Makefile` - Build automation
- [ ] Create `pyproject.toml` - Project metadata
- [ ] Verify `requirements.txt` - Python dependencies (already exists)
- [ ] Create `.gitignore` - Version control exclusions

---

## 🔧 Python Environment Setup

- [ ] Create virtual environment (optional but recommended)
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  ```

- [ ] Set PYTHONPATH for imports from src/
  ```bash
  set PYTHONPATH=c:\Users\Luan\Desktop\Data Science
  ```

- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

---

## 📊 Model Setup

### Update Notebook Code
- [ ] Modify `notebooks/03_modeling.ipynb` last cell to save models to `models/v1/`:
  ```python
  MODELS_DIR = os.path.join(os.getcwd(), 'models', 'v1')
  os.makedirs(MODELS_DIR, exist_ok=True)
  
  joblib.dump(gb_model, os.path.join(MODELS_DIR, 'model.pkl'))
  joblib.dump(scaler_standard, os.path.join(MODELS_DIR, 'scaler.pkl'))
  joblib.dump(preprocessing_config, os.path.join(MODELS_DIR, 'metadata.json'))
  ```

### Run Training Pipeline
- [ ] Run `notebooks/01_eda.ipynb` - Exploratory analysis
- [ ] Run `notebooks/02_feature_engineering.ipynb` - Feature engineering
- [ ] Run `notebooks/03_modeling.ipynb` - Model training and save

### Verify Models Saved
- [ ] Check `models/v1/model.pkl` exists
- [ ] Check `models/v1/scaler.pkl` exists
- [ ] Check `models/v1/metadata.json` exists

---

## 🚀 Testing

### Unit Tests
- [ ] Run feature tests: `pytest tests/test_features.py`
- [ ] Run model tests: `pytest tests/test_model.py`
- [ ] Run API tests: `pytest tests/test_api.py`

### Integration Testing
- [ ] Test model loader imports correctly
  ```python
  from app.model_loader import ModelDeployment
  deployment = ModelDeployment()
  deployment.load_all()
  ```

### Dashboard Testing
- [ ] Launch dashboard: `streamlit run app/app.py`
- [ ] Test prediction tab
- [ ] Test analytics tab
- [ ] Test about tab
- [ ] Test different input values

---

## 🐳 Deployment

### Local Testing
- [ ] Test streamlit command
- [ ] Test with custom port
- [ ] Test error handling
- [ ] Test module imports

### Docker (Optional)
- [ ] Build Docker image: `docker build -t churn-prediction .`
- [ ] Run Docker container: `docker run -p 8501:8501 churn-prediction`
- [ ] Verify Docker deployment

### Makefile (Optional)
- [ ] Test `make install`
- [ ] Test `make train`
- [ ] Test `make test`
- [ ] Test `make dashboard`

---

## 📚 Documentation

### Code Documentation
- [ ] Add docstrings to all Python files
- [ ] Add type hints to function signatures
- [ ] Add inline comments for complex logic

### User Documentation
- [ ] Update README.md with new structure
- [ ] Update DEPLOYMENT_GUIDE.md (already done ✅)
- [ ] Update QUICK_REFERENCE.txt (already done ✅)
- [ ] Create ARCHITECTURE.md (optional)

---

## ✅ Final Verification

- [ ] All directories created and accessible
- [ ] All files migrated to correct locations
- [ ] All imports work from app/ and src/
- [ ] PYTHONPATH configured correctly
- [ ] Models saved in models/v1/
- [ ] Requirements.txt installed
- [ ] Streamlit dashboard runs: `streamlit run app/app.py`
- [ ] Dashboard accessible at http://localhost:8501
- [ ] Predictions working correctly
- [ ] No import errors
- [ ] All tests passing

---

## 📋 Optional Enhancements

- [ ] Set up Git hooks (pre-commit)
- [ ] Configure GitHub Actions workflows
- [ ] Add model versioning/registry
- [ ] Add API endpoint tests
- [ ] Add performance monitoring
- [ ] Add logging to all modules
- [ ] Create development guide
- [ ] Create API documentation

---

## 🎯 Success Criteria

✅ All directories created  
✅ All files migrated to correct locations  
✅ All Python modules importable  
✅ Streamlit dashboard runs without errors  
✅ Model predictions working correctly  
✅ All tests passing  
✅ Documentation complete and updated  

---

**Estimated Implementation Time:** 2-3 hours  
**Difficulty Level:** Medium  
**Dependencies:** Python 3.7+, pip packages from requirements.txt

---

## 📞 Troubleshooting Tips

### ModuleNotFoundError: No module named 'src'
→ Set PYTHONPATH: `set PYTHONPATH=%cd%`

### FileNotFoundError: data not found
→ Check files are in `data/raw/` directory

### Models not loading
→ Verify `models/v1/` directory and files exist

### Streamlit import error
→ Run: `pip install -r requirements.txt`

### Port 8501 already in use
→ Use: `streamlit run app/app.py --server.port 8502`

---

**Status:** Ready for Implementation  
**Last Updated:** January 28, 2026
