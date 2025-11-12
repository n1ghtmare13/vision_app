# /vision_app/vision_app/config.py
# Contains all configuration constants for the application.

import os

# --- General App Settings ---
APP_NAME_GUI = "Vision App GUI"
APP_NAME_CONSOLE = "Vision App - Console"

# --- Paths ---
# Build paths relative to the project root to ensure portability
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH_XML = os.path.join(MODELS_DIR, "yolo11n_object365.xml")

# --- Inference Settings ---
DEVICE_NAME = "CPU"
CACHE_DIR = "cache"

# --- Camera Settings ---
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# --- Model Parameters ---
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
