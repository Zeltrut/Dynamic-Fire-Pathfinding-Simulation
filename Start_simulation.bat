@echo off
title Dynamic Fire Pathfinding Setup

echo ========================================================
echo  Checking System Configuration...
echo ========================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not found in PATH.
    echo Please install Python from python.org and try again.
    pause
    exit
)

:: 2. Check if Virtual Environment exists, if not then create it
if not exist venv (
    echo Virtual Environment not found. Creating one now...
    echo This may take a moment...
    python -m venv venv
)

:: 3. Activate that Virtual Environment
call venv\Scripts\activate

:: 4. Install Dependencies automatically
echo.
echo Checking for required libraries (Pygame, Numpy)...
pip install -r requirements.txt >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo Installing libraries failed. Trying to force install...
    pip install pygame numpy pandas tqdm ipython
)

:: 5. Run the Simulation
echo.
echo ========================================================
echo  Starting Simulation...
echo ========================================================
python run_simulation.py

:: 6. Pause so they can see any errors if it crashes
pause