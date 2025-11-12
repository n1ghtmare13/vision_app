# /vision_app/vision_app/ui_cv2/drawing.py
# Drawing functions for the OpenCV window.

import cv2
from ..labels import CLASSES
from ..utils import generate_random_colors

COLORS = generate_random_colors(len(CLASSES))


def draw_results(frame, results):
    """Draws bounding boxes and labels on the frame."""
    for box, cls_id, conf, track_id in results:
        x1, y1, x2, y2 = box
        color = [int(c) for c in COLORS[cls_id]]
        label = f"ID:{track_id} {CLASSES[cls_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
