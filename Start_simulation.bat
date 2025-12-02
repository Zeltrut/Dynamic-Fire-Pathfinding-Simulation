@echo off
echo Starting Fire Simulation...

:: activate the virtual environment
call venv\Scripts\activate

:: run the simulation script
python run_simulation.py

:: pause so you can see any error messages before closing
pause