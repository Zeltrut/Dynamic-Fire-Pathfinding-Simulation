import sys
import os

# Ensure the script can see the subfolders
sys.path.append(os.getcwd())

from Frontend_logic.visualizer import FireSimulationApp

if __name__ == "__main__":
    app = FireSimulationApp()
    app.run()