from ultralytics import YOLO

MODEL_NAME = "yolo11n_object365"

# Load the model (n - nano, the small one)
# On the first run, the model will be automatically downloaded
print(f"Loading {MODEL_NAME}.pt model...")
model = YOLO(f"{MODEL_NAME}.pt")

# Export model to ONNX format
print("Exporting to ONNX format...")
model.export(format="onnx", opset=12)

print(f"Export complete. File {MODEL_NAME}.onnx has been created.")
