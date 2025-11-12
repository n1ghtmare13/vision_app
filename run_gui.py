# /run_gui.py
# Entry point for the Dear PyGui application.

from vision_app import CameraManager, InferenceEngine
from vision_app.ui_dpg import AppGui


def main():
    """Initializes and runs the GUI application."""
    camera = CameraManager(backend="ffmpeg")
    engine = InferenceEngine(camera)

    # The GUI needs the list of available cameras to create the dropdown
    available_cams = camera.find_available_cameras_ffmpeg()
    if not available_cams:
        print(
            "Error: No cameras found. Please connect a camera and ensure FFmpeg is installed."
        )
        return

    gui = AppGui(camera, engine, available_cams)

    try:
        # Start the backend threads
        camera.start(camera_id_or_name=available_cams[0])
        engine.start()

        # Run the GUI loop (this will block until the window is closed)
        gui.run()

    finally:
        print("Shutting down...")
        engine.stop()
        camera.stop()


if __name__ == "__main__":
    main()
