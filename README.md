# Dynamic-Fire-Pathfinding-Simulation

## Overview
Our project simulates a building on fire, with available paths dynamically altered. We compare **A* (A-Star)** as an informed search and **BFS (Breadth-First Search)** as a uniform baseline to see which finds a better path from start to goal as the grid changes.

**PyGame** is used for visualization to showcase the different danger levels, fire intensity, and agent movement. By evaluating each algorithm's performance through multiple simulations using random seeds, we can measure:
* **Success Rate**
* **Number of Steps**
* **Danger Exposure**
These metrics help to determine which method adapts better to a dynamic, and hazardous environment.

## How to Run (Windows ONLY), for MAC must install dependencies...

### **Option 1: Executable (Recommended)**
Navigate to the **`WINDOWS`** folder and double click on: **`FireSim.exe`**

### **Option 2: Manual (If you have all the dependencies)**
If you prefer to run the project manually or use the launcher script, simply run: **`Start_simulation.bat`** OR **`python run_simulation.py`**


## Contributers

* **Giselle:** Grid creation (labels), penalties/rewards cost, fire implementation
* **Tyler:** algorithm implementations (BFS and A*), metric collection (time, success rate, mean steps to goal, mean "danger exposure"), character implementation
* **Justin:** backbone of visualizer 
* **Sara:** frontend of visualizer
