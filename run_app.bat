@echo off
REM OmniShopAgent Startup Script for Windows

cd /d "%~dp0"

if not exist "venv\" (
    echo Virtual environment not found!
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting OmniShopAgent...
echo App will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.

python -m streamlit run app.py

