# /vision_app/vision_app/ui_cv2/main_window.py
# Main display loop for the OpenCV window.

import cv2
import time
from .drawing import draw_results
from .. import config


def run_viewer(camera, engine):
    """Creates a window and displays frames with detection results."""
    display_fps, frame_count, start_time = 0, 0, time.time()

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        results = engine.get_results()
        draw_results(frame, results)

        # Calculate and display FPS metrics
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        cv2.putText(
            frame,
            f"Display FPS: {display_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Inference FPS: {engine.fps:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow(config.APP_NAME_CONSOLE, frame)

        # Exit on 'q' or window close
        if (
            cv2.waitKey(1) & 0xFF == ord("q")
            or cv2.getWindowProperty(config.APP_NAME_CONSOLE, cv2.WND_PROP_VISIBLE) < 1
        ):
            break

    cv2.destroyAllWindows()
