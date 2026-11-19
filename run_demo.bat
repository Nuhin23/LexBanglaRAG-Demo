@echo off
title LexBanglaRAG - Interactive Demo Launcher
color 0b
echo =====================================================================
echo           LexBanglaRAG: Bengali Statutory Legal AI Demo
echo     Department of Software Engineering, Daffodil International University
echo =====================================================================
echo.

echo Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH! Please install Python 3.10+
    pause
    exit /b
)

echo.
echo Installing required dependencies (Streamlit, Pandas)...
pip install -r requirements.txt

echo.
echo =====================================================================
echo Starting Streamlit Web App...
echo (It will automatically open in your default browser at http://localhost:8501)
echo Press Ctrl+C in this window anytime to stop the server.
echo =====================================================================
echo.
python -m streamlit run app.py

pause
