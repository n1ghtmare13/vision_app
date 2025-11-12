# /vision_app/vision_app/ui_dpg/drawing.py
# Drawing functions for the Dear PyGui interface.

import dearpygui.dearpygui as dpg
from ..labels import CLASSES
from ..utils import generate_readable_colors
from .. import config

COLORS = generate_readable_colors(len(CLASSES))


def draw_results(results, parent_layer):
    """Draws detection results on a DPG draw layer."""
    for box, cls_id, conf, track_id in results:
        if cls_id < len(CLASSES):
            x1, y1, x2, y2 = map(int, box)

            color = tuple(COLORS[cls_id]) + (255,)
            label = f"ID:{track_id} {CLASSES[cls_id]}: {conf:.2f}"

            dpg.draw_rectangle(
                (x1, y1), (x2, y2), color=color, thickness=3, parent=parent_layer
            )

            # --- Draw text with background for readability ---
            label_height = 20
            padding = 5
            text_y = y1 - label_height - padding
            if text_y < 0:
                text_y = y1 + padding

            text_bg_color = (255, 255, 255, 180)
            text_color = (0, 0, 0, 255)

            approx_char_width = label_height * 0.55
            bg_width = int(len(label) * approx_char_width)
            bg_height = label_height + 4
            bg_x2 = min(x1 + bg_width + 6, config.CAM_WIDTH)
            bg_y2 = min(text_y + bg_height, config.CAM_HEIGHT)
            box_thickness = 3

            dpg.draw_rectangle(
                (max(0, x1 + box_thickness), max(0, text_y - 2)),
                (bg_x2, bg_y2),
                color=text_bg_color,
                fill=text_bg_color,
                parent=parent_layer,
            )
            dpg.draw_text(
                (x1 + box_thickness + 2, text_y),
                label,
                color=text_color,
                size=label_height,
                parent=parent_layer,
            )
