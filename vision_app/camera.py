# /vision_app/vision_app/camera.py
# Manages video capture from different backends (cv2, ffmpeg) in a separate thread.

import threading
import subprocess as sp
import re
import time
import cv2
import numpy as np
from . import config


class CameraManager:
    """A thread-safe class to manage camera access."""

    def __init__(self, backend="ffmpeg"):
        self.backend = backend
        self.frame = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active_camera = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process = None  # For ffmpeg subprocess
        self._cap = None  # For OpenCV VideoCapture object

    def start(self, camera_id_or_name=0):
        """Starts the camera capture thread."""
        self.active_camera = camera_id_or_name
        self._thread.start()

    def stop(self):
        """Stops the camera capture thread and releases resources."""
        self.stop_event.set()
        self._thread.join(timeout=2)
        if self._process and self._process.poll() is None:
            self._process.kill()
        if self._cap:
            self._cap.release()

    def get_frame(self):
        """Returns the latest captured frame in a thread-safe manner."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def set_camera(self, camera_name):
        """Changes the active camera. The capture loop will handle the switch."""
        print(f"Switching camera to: {camera_name}")
        with self.frame_lock:
            self.active_camera = camera_name

    @staticmethod
    def find_available_cameras_ffmpeg():
        """Uses FFmpeg to find available cameras on Windows (dshow backend)."""
        command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        try:
            result = sp.run(command, capture_output=True, text=True, check=False)
            output = result.stderr
        except FileNotFoundError:
            print("Error: FFmpeg is not installed or not in the system PATH.")
            return []
        pattern = re.compile(r'"([^"]+)" \(video\)')
        return pattern.findall(output)

    def _capture_loop(self):
        """Main loop for the thread, dispatches to the correct backend loop."""
        if self.backend == "ffmpeg":
            self._ffmpeg_loop()
        else:
            self._cv2_loop()

    def _cv2_loop(self):
        """Capture loop using OpenCV."""
        while not self.stop_event.is_set():
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.active_camera, cv2.CAP_MSMF)
                if not self._cap.isOpened():
                    print(
                        f"Error: Cannot open camera {self.active_camera} with OpenCV."
                    )
                    time.sleep(1)
                    continue
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self._cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
            else:
                # If reading fails, release and try to reconnect
                self._cap.release()
                self._cap = None
                time.sleep(0.5)

    def _ffmpeg_loop(self):
        """Capture loop using FFmpeg subprocess for robust camera switching."""
        current_camera = None
        while not self.stop_event.is_set():
            with self.frame_lock:
                target_camera = self.active_camera

            if target_camera != current_camera:
                if self._process and self._process.poll() is None:
                    print(f"Stopping camera: {current_camera}")
                    self._process.kill()
                    self._process.wait()

                if target_camera:
                    print(f"Starting camera: {target_camera}")
                    command = [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-f",
                        "dshow",
                        "-video_size",
                        f"{config.CAM_WIDTH}x{config.CAM_HEIGHT}",
                        "-framerate",
                        "30",
                        "-i",
                        f"video={target_camera}",
                        "-pix_fmt",
                        "bgr24",
                        "-f",
                        "rawvideo",
                        "-",
                    ]
                    self._process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL)
                    current_camera = target_camera
                else:
                    self._process = None
                    current_camera = None

            if self._process and self._process.poll() is None:
                raw_frame = self._process.stdout.read(
                    config.CAM_WIDTH * config.CAM_HEIGHT * 3
                )
                if len(raw_frame) == config.CAM_WIDTH * config.CAM_HEIGHT * 3:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (config.CAM_HEIGHT, config.CAM_WIDTH, 3)
                    )
                    with self.frame_lock:
                        self.frame = frame
                else:
                    print(f"Stream from '{current_camera}' interrupted, restarting...")
                    self._process.kill()
                    self._process = None
                    current_camera = None
                    time.sleep(1)
            else:
                time.sleep(0.1)

        if self._process and self._process.poll() is None:
            self._process.kill()
        print("Camera thread finished.")
