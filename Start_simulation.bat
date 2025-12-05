@echo off
title Dynamic Fire Simulation Setup
echo ==========================================
echo  Setting up Dynamic Fire Pathfinding...
echo ==========================================

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install it to run this simulation.
    pause
    exit
)

:: 2. Create Virtual Environment (if missing)
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: 3. Install Dependencies
echo Installing requirements...
call venv\Scripts\activate
pip install -r requirements.txt >nul 2>&1

:: 4. Run the Simulation
echo.
echo Starting Simulation...
python run_simulation.py

deactivate
pause