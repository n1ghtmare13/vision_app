# /vision_app/vision_app/utils.py
# Contains utility functions used across the application.

import colorsys
import numpy as np


def generate_readable_colors(n):
    """Generate n distinct, saturated colors for good visibility (used by DPG GUI)."""
    colors = []
    golden_ratio = 0.61803398875
    hue = 0.0
    for i in range(n):
        hue = (hue + golden_ratio) % 1.0
        saturation = 1.0
        value = 0.7
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(255 * c) for c in rgb))
    return np.array(colors, dtype=np.uint8)


def generate_random_colors(n):
    """Generate n random colors (used by OpenCV UI)."""
    np.random.seed(42)
    return np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)
