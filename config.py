import os

MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

# Teljesítmény optimalizálás
FRAME_SKIP_N = 3           # Csak minden N.-edik képkockát dolgozza fel
RESIZE_WIDTH = 640         # Kép átméretezése a gyorsabb detektáláshoz
RESIZE_HEIGHT = 480

GRID_COLS = 10
GRID_ROWS = 10

# Region of Interest (ROI) - kért tartomány arányai (0.0 - 1.0)
# Példa: Csak a kép alsó 70%-át és középső részét vizsgálja (út/pálya)
USE_ROI = True
ROI_Y_MIN = 0.3
ROI_Y_MAX = 1.0
ROI_X_MIN = 0.0
ROI_X_MAX = 1.0