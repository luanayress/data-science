# 🔧 Quick Setup Commands Reference

**Date:** January 28, 2026  
**Project:** Customer Churn Prediction  
**Purpose:** One-stop reference for all setup commands

---

## 📂 Phase 1: Create Directory Structure

```bash
# Navigate to project
cd "c:\Users\Luan\Desktop\Data Science"

# Create all directories
mkdir app
mkdir src\data
mkdir src\features
mkdir src\models
mkdir src\pipelines
mkdir src\utils
mkdir notebooks
mkdir models\v1
mkdir data\raw
mkdir data\processed
mkdir data\external
mkdir tests
mkdir configs
mkdir .github\workflows

# Verify structure
dir /s /b
```

---

## 📋 Phase 2: Move & Rename Existing Files

```bash
# Notebooks (rename with numeric prefix)
move exploratory_analysis.ipynb notebooks\01_eda.ipynb
move feature_eng.ipynb notebooks\02_feature_engineering.ipynb
move modeling.ipynb notebooks\03_modeling.ipynb

# Data files
move Customer-Churn-Records.csv data\raw\
move Customer_Churn_Final_Features.csv data\processed\
move Customer_Churn_Engineered_Features.csv data\processed\

# Keep documentation files in root
# (DEPLOYMENT_GUIDE.md, README.md, etc.)
```

---

## 🐍 Phase 3: Python Environment Setup

```bash
# Set Python path for module imports
set PYTHONPATH=c:\Users\Luan\Desktop\Data Science

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
pip list | grep scikit-learn
```

---

## 📝 Phase 4: Create Core Python Files

### Create app/ files
```bash
# app/__init__.py (empty)
type nul > app\__init__.py

# app/app.py (copy from old app.py, update imports)
# app/model_loader.py (copy from model_deployment.py, update paths)
# app/schema.py (create new with Pydantic models)
```

### Create src/ files
```bash
# Create __init__.py in all directories
type nul > src\__init__.py
type nul > src\data\__init__.py
type nul > src\features\__init__.py
type nul > src\models\__init__.py
type nul > src\pipelines\__init__.py
type nul > src\utils\__init__.py
type nul > tests\__init__.py

# Create Python modules (see IMPLEMENTATION_CHECKLIST.md for details)
# src/data/load_data.py
# src/features/build_features.py
# src/models/train.py
# etc.
```

---

## 🚀 Phase 5: Run Training & Save Models

```bash
# Navigate to project
cd "c:\Users\Luan\Desktop\Data Science"

# Set Python path
set PYTHONPATH=%cd%

# Run notebooks in order
jupyter notebook notebooks\01_eda.ipynb
jupyter notebook notebooks\02_feature_engineering.ipynb
jupyter notebook notebooks\03_modeling.ipynb

# In 03_modeling.ipynb, last cell should contain:
# (See DEPLOYMENT_GUIDE.md Step 2 for exact code)

# This creates:
# - models\v1\model.pkl
# - models\v1\scaler.pkl
# - models\v1\metadata.json

# Verify models saved
dir models\v1
```

---

## ✅ Phase 6: Testing

```bash
# Test Python path and imports
python -c "from app.model_loader import ModelDeployment; print('✓ Import successful')"

# Test model loading
python -c "from app.model_loader import ModelDeployment; m = ModelDeployment(); m.load_all(); print('✓ Models loaded')"

# Run unit tests (if created)
pytest tests\ -v

# Test Streamlit command syntax
streamlit run app\app.py --help
```

---

## 🌐 Phase 7: Launch Dashboard

```bash
# Basic launch
streamlit run app\app.py

# Launch with custom port (if 8501 is busy)
streamlit run app\app.py --server.port 8502

# Launch with verbose logging
streamlit run app\app.py --logger.level=debug

# Access in browser
http://localhost:8501
```

---

## 🐳 Phase 8: Docker (Optional)

```bash
# Build Docker image
docker build -t churn-prediction .

# Run Docker container
docker run -p 8501:8501 churn-prediction

# Check running containers
docker ps

# Stop container
docker stop <container_id>
```

---

## 🔨 Phase 9: Makefile (Optional)

```bash
# Create Makefile and run targets
make install      # Install dependencies
make train        # Run training pipeline
make test         # Run tests
make dashboard    # Start Streamlit dashboard
make docker-build # Build Docker image
make docker-run   # Run Docker container
```

---

## 🧪 Testing Commands

```bash
# Quick health check
python -c "import streamlit; import sklearn; import pandas; print('✓ All imports OK')"

# Check model files
dir models\v1
python -c "import joblib; model = joblib.load('models/v1/model.pkl'); print('✓ Model loads OK')"

# List all Python files
python -c "import os; [print(f) for f in os.walk('src') for file in os.listdir(os.path.join(f[0])) if file.endswith('.py')]"
```

---

## 📊 Verification Checklist Commands

```bash
# 1. Check directory structure
tree /L 3

# 2. Check Python path
echo %PYTHONPATH%

# 3. Check dependencies
pip list | grep -E "streamlit|sklearn|pandas"

# 4. Check notebooks
dir notebooks\*.ipynb

# 5. Check data files
dir data\raw\*.csv
dir data\processed\*.csv

# 6. Check models
dir models\v1\*

# 7. Check Python modules
tree src /L 2

# 8. Check test files
dir tests\*.py

# 9. Quick import test
python -c "from src.data.load_data import load_raw_data; print('✓ OK')"

# 10. Dashboard launch test
streamlit run app\app.py --headless &
```

---

## 🔄 Common Troubleshooting Commands

```bash
# Clear Python cache
python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version

# Verify pip is up to date
python -m pip install --upgrade pip

# Clear pip cache
pip cache purge

# Find file in project
dir /s /b "*filename*"

# Check port usage
netstat -ano | findstr :8501

# Kill process on port
taskkill /PID <process_id> /F
```

---

## 🎯 Fast Implementation (Copy-Paste)

### All-in-one setup (run these commands in sequence):

```bash
# 1. Navigate
cd "c:\Users\Luan\Desktop\Data Science"

# 2. Create directories
mkdir app src\data src\features src\models src\pipelines src\utils notebooks models\v1 data\raw data\processed data\external tests configs .github\workflows

# 3. Move files
move exploratory_analysis.ipynb notebooks\01_eda.ipynb
move feature_eng.ipynb notebooks\02_feature_engineering.ipynb
move modeling.ipynb notebooks\03_modeling.ipynb
move Customer-Churn-Records.csv data\raw\
move Customer_Churn_Final_Features.csv data\processed\
move Customer_Churn_Engineered_Features.csv data\processed\

# 4. Create __init__ files
type nul > app\__init__.py
type nul > src\__init__.py
type nul > src\data\__init__.py
type nul > src\features\__init__.py
type nul > src\models\__init__.py
type nul > src\pipelines\__init__.py
type nul > src\utils\__init__.py
type nul > tests\__init__.py

# 5. Set Python path
set PYTHONPATH=%cd%

# 6. Install dependencies
pip install -r requirements.txt

# 7. Ready for model training
echo "✓ Structure ready! Run notebooks in order:"
echo "  1. jupyter notebook notebooks\01_eda.ipynb"
echo "  2. jupyter notebook notebooks\02_feature_engineering.ipynb"
echo "  3. jupyter notebook notebooks\03_modeling.ipynb"

# 8. After training, launch dashboard
echo "✓ Models ready! Launch dashboard:"
echo "  streamlit run app\app.py"
```

---

## 📚 Documentation Commands

```bash
# View setup summary
type STRUCTURE_UPDATE_SUMMARY.md

# View implementation checklist
type IMPLEMENTATION_CHECKLIST.md

# View quick reference
type QUICK_REFERENCE.txt

# View deployment guide
type DEPLOYMENT_GUIDE.md

# View all MD files
dir /b *.md
```

---

## 🔗 Important Notes

1. **PYTHONPATH Setup**
   ```bash
   set PYTHONPATH=c:\Users\Luan\Desktop\Data Science
   # Required for: from src.* import ...
   ```

2. **Model Saving Code** (for notebooks/03_modeling.ipynb last cell)
   - See DEPLOYMENT_GUIDE.md STEP 2
   - Saves to models/v1/ directory

3. **Port Issues**
   - If 8501 is busy: `streamlit run app\app.py --server.port 8502`

4. **Module Import Errors**
   - Always set PYTHONPATH before running
   - Check __init__.py files exist in all directories

5. **First Run**
   - May take longer due to Streamlit compilation
   - Check http://localhost:8501 after message appears

---

**Last Updated:** January 28, 2026  
**Status:** Ready for Use  
**Reference:** See IMPLEMENTATION_CHECKLIST.md for detailed steps
