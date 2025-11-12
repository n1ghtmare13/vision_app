# /vision_app/vision_app/vision.py
# Handles model inference and object tracking in a separate thread.

import threading
import time
import openvino as ov
import numpy as np
import cv2
from boxmot import ByteTrack
from . import config
from .labels import CLASSES


class InferenceEngine:
    """A thread-safe class for running model inference."""

    def __init__(self, camera_manager):
        self.camera = camera_manager
        self.results = []
        self.results_lock = threading.Lock()
        self.fps = 0
        self.stop_event = threading.Event()

        # Initialize OpenVINO Core and compile the model
        core = ov.Core()
        model = core.read_model(config.MODEL_PATH_XML)
        self.compiled_model = core.compile_model(
            model, config.DEVICE_NAME, {"CACHE_DIR": config.CACHE_DIR}
        )
        self.output_layer = self.compiled_model.output(0)

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)

    def start(self):
        """Starts the inference thread."""
        self._thread.start()

    def stop(self):
        """Stops the inference thread."""
        self.stop_event.set()
        self._thread.join(timeout=2)

    def get_results(self):
        """Returns the latest inference results in a thread-safe manner."""
        with self.results_lock:
            return self.results.copy()

    def _inference_loop(self):
        """Main loop for the inference thread."""
        tracker = ByteTrack(per_class=True, nr_classes=len(CLASSES), verbose=False)
        frame_count, start_time = 0, time.time()

        while not self.stop_event.is_set():
            frame_to_process = self.camera.get_frame()
            if frame_to_process is None:
                time.sleep(0.01)
                continue

            # Pre-process the frame for the model
            frame_height, frame_width = frame_to_process.shape[:2]
            scale = min(
                config.INPUT_WIDTH / frame_width, config.INPUT_HEIGHT / frame_height
            )
            scaled_w, scaled_h = int(frame_width * scale), int(frame_height * scale)
            resized_frame = cv2.resize(frame_to_process, (scaled_w, scaled_h))
            padded_frame = np.full(
                (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3), 114, dtype=np.uint8
            )
            padded_frame[:scaled_h, :scaled_w] = resized_frame
            input_tensor = (
                np.expand_dims(padded_frame, 0).transpose(0, 3, 1, 2).astype(np.float32)
                / 255.0
            )

            # Run inference
            result = self.compiled_model([input_tensor])[self.output_layer]
            raw_detections = result.transpose(0, 2, 1)[0]

            # Post-process detections and apply Non-Maximum Suppression (NMS)
            boxes, confidences, class_ids = [], [], []
            for det in raw_detections:
                scores = det[4:]
                conf = np.max(scores)
                if conf > config.CONFIDENCE_THRESHOLD:
                    cls_id = np.argmax(scores)
                    cx, cy, w, h = det[:4]
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    boxes.append([x1, y1, int(w), int(h)])
                    confidences.append(float(conf))
                    class_ids.append(cls_id)

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD
            )

            # Prepare detections for the tracker
            detections_for_tracker = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections_for_tracker.append(
                        [x, y, x + w, y + h, confidences[i], class_ids[i]]
                    )

            # Update the tracker
            tracked_objects = tracker.update(
                np.array(detections_for_tracker), frame_to_process
            )

            # Store final results, scaling coordinates back to original frame size
            current_results = []
            if len(tracked_objects) > 0:
                for x1, y1, x2, y2, track_id, conf, class_id, _ in tracked_objects:
                    x1_s, y1_s = int(x1 / scale), int(y1 / scale)
                    x2_s, y2_s = int(x2 / scale), int(y2 / scale)
                    current_results.append(
                        [(x1_s, y1_s, x2_s, y2_s), int(class_id), conf, int(track_id)]
                    )

            with self.results_lock:
                self.results = current_results

            # Calculate inference FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count, start_time = 0, time.time()

        print("Inference thread finished.")
