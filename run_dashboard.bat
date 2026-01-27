@echo off
REM ==========================================
REM Quick Start Script for Churn Prediction Dashboard
REM ==========================================
REM This batch file helps run the Streamlit dashboard
REM
REM Usage:
REM   1. Save this file as 'run_dashboard.bat' in project directory
REM   2. Double-click the file to run
REM   3. Browser will open automatically
REM
REM Author: Data Science Team
REM Date: 2026-01-27
REM ==========================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo Customer Churn Prediction Dashboard
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found: 
python --version
echo.

REM Check if Streamlit is installed
python -m pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/4] Installing dependencies...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [2/4] Dependencies already installed
)
echo.

REM Check if models directory exists
if not exist "models" (
    echo ERROR: models/ directory not found!
    echo.
    echo Please make sure you have:
    echo   1. Run feature_eng.ipynb
    echo   2. Run modeling.ipynb
    echo   3. Saved models (see DEPLOYMENT_GUIDE.md)
    echo.
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "models\gradient_boosting_model.pkl" (
    echo ERROR: gradient_boosting_model.pkl not found!
    echo Please save models first (see DEPLOYMENT_GUIDE.md)
    pause
    exit /b 1
)

echo [3/4] Model artifacts verified
echo   - gradient_boosting_model.pkl found
echo   - scaler_standard.pkl found
echo   - scaler_minmax.pkl found
echo   - preprocessing_config.pkl found
echo.

echo [4/4] Starting Streamlit dashboard...
echo.
echo ============================================================
echo Dashboard is starting...
echo Local URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ============================================================
echo.

REM Launch Streamlit
python -m streamlit run app.py

pause
