# /run_console.py
# Entry point for the console (OpenCV window) application.

from vision_app import CameraManager, InferenceEngine
from vision_app.ui_cv2 import run_viewer


def main():
    """Initializes and runs the console application."""
    camera = CameraManager(backend="cv2")
    engine = InferenceEngine(camera)

    try:
        # Start the backend threads with the default camera (ID 0)
        camera.start(camera_id_or_name=0)
        engine.start()

        # Run the OpenCV display loop (this will block until 'q' is pressed)
        run_viewer(camera, engine)

    finally:
        print("Shutting down...")
        engine.stop()
        camera.stop()


if __name__ == "__main__":
    main()
