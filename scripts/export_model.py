# /scripts/export_model.py
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_NAME = "yolo11n_object365"  # Model you have or want to download

# --- PATH SETUP ---
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

# --- LOAD MODEL ---
pt_path = os.path.join(models_dir, f"{MODEL_NAME}.pt")

# --- If .pt file exists, load it; otherwise, download the model ---
model_source = pt_path if os.path.exists(pt_path) else MODEL_NAME
print(f"Loading model from: {model_source}")
model = YOLO(model_source)

# --- EXPORT TO ONNX ---
print("Exporting to ONNX format...")
onnx_file = f"{MODEL_NAME}.onnx"
model.export(format="onnx", opset=12)  # creates MODEL_NAME.onnx in current directory

# --- MOVE ONNX TO models_dir ---
for file_name in [f"{MODEL_NAME}.pt", f"{MODEL_NAME}.onnx"]:
    src = os.path.join(os.getcwd(), file_name)
    dst = os.path.join(models_dir, file_name)
    if os.path.exists(src) and src != dst:
        os.rename(src, dst)
        print(f"Moved '{file_name}' to '{models_dir}'")

print("Script finished successfully.")
