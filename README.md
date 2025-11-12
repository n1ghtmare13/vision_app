# Real-Time Object Detection Application

This repository contains a real-time object detection application developed in Python. It utilizes a YOLOv11n model trained on the Objects365 dataset, processed with OpenVINO for high-performance inference on Intel CPUs and iGPUs. The application features a multithreaded architecture for smooth video rendering and a `dearpygui`-based user interface.

## Features

- **High-Performance Inference:** Powered by OpenVINO for optimized performance on Intel hardware.
- **Advanced Object Model:** Uses a YOLOv11n model trained on the extensive **Objects365 dataset** (365 classes).
- **Real-Time Tracking:** Implements the `ByteTrack` algorithm via the `boxmot` library for stable and consistent object tracking.
- **Asynchronous Architecture:** A multithreaded design separates video capture, inference, and rendering to ensure a smooth UI experience (target 30 FPS display) regardless of inference speed.
- **Flexible Camera Handling:** Automatically detects all cameras connected to the system (using FFmpeg) and allows switching between them at runtime.
- **Modern GUI:** Built with `dearpygui` for a responsive and performant user interface.

## Prerequisites

Before you begin, ensure you have the following installed on your system (developed and tested on Windows):

- [Python 3.9–3.12](https://www.python.org/)
  (requires Python ≥3.9 and ≤3.12)
- [Git](https://git-scm.com/)
- [FFmpeg](https://ffmpeg.org/download.html)
  - **IMPORTANT:** Ensure the directory containing `ffmpeg.exe` is added to your system's `PATH` environment variable. You can verify this by running `ffmpeg -version` in a new terminal window.

## Developer Setup

Follow these steps to set up the development environment.

### 1. Clone the Repository
```powershell
git clone https://github.com/n1ghtmare13/vision_app.git
cd vision_app
```

### 2. Create and Activate a Virtual Environment
This is crucial to keep project dependencies isolated.
```powershell
# Create a virtual environment named 'venv'
python -m venv venv
# Activate the virtual environment
.\venv\Scripts\activate
```
*(On Windows, your terminal prompt should now start with `(venv)`).*

*Remember to activate the virtual environment each time you work on the project.*


### 3. Configure Pip Version (Important!)
For compatibility with our development tools (`pip-tools`), a specific version of `pip` is required.
```powershell
# This command will downgrade pip if necessary
.\venv\Scripts\python.exe -m pip install "pip<25"
```

### 4. Install Python Dependencies
This project uses `pip-tools` for precise dependency management. The following commands will install all necessary packages for running the application and for development.
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
### 5. Run the Application
You are now ready to run the application. The required models (.xml and .bin) are already located in the `models/` directory.
- **To run the GUI version with Dear PyGui (recommended):**
  ```powershell
  python run_gui.py
  ```
- **To run the simple console version with an OpenCV window:**
  ```powershell
  python run_console.py
  ```

---

## Model Generation (Advanced)

The repository includes pre-converted OpenVINO models in the `models/` directory. You only need to follow these steps if you want to **re-generate** or **update** the model files.

**a) Download the Core Model**

You can obtain models in two ways:

1. **Using a YOLOv11n model trained on Objects365:**
   This model must be downloaded manually.

   * **Download link:** [yolo11n_object365.pt](https://huggingface.co/NRtred/yolo11n_object365/resolve/main/yolo11n_object365.pt?download=true)
   * **Save it** in the `models/` directory of the project.

2. **Using an official model trained on COCO:**
   Set `MODEL_NAME = "DESIRED_MODEL"` in `scripts/export_model.py` to match an official model available in the `ultralytics` package (e.g., `"yolov10n"`), and then run the script to download it automatically.

**b) Convert the Model to OpenVINO Format**

The `ovc` (OpenVINO Converter) tool is required for this step (installed via `requirements.txt`).

```powershell
# First, convert the PyTorch model (.pt) to ONNX format
python scripts/export_model.py

# Then, convert the ONNX model to OpenVINO IR format (.xml + .bin)
ovc models/yolo11n_object365.onnx --output_model models/yolo11n_object365.xml
```

## Dependency Management Workflow

This project uses `pip-tools` to manage dependencies. This ensures that all developers work with the exact same package versions.

**To add or update a dependency:**

1.  **Install `pip-tools`** (if not already installed via `requirements-dev.txt`):
    ```powershell
    pip install "pip-tools"
    ```
2.  **Edit the source file:**
    - For core application dependencies, add the package name to `requirements.in`.
    - For development-only tools, add the package name to `requirements-dev.in`.

3.  **Re-compile the dependency files:**
    ```powershell
    pip-compile requirements.in
    pip-compile requirements-dev.in
    ```
    *This will update the corresponding `.txt` files with the new package and all its sub-dependencies.*

4.  **Synchronize your environment:**
    ```powershell
    pip-sync requirements.txt requirements-dev.txt
    ```
    *This command will install/uninstall packages to perfectly match the state defined in the `.txt` files.*

5.  **Commit both the `.in` and `.txt` files** to the repository to share the updated dependencies with the team.
