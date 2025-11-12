# vision_app/vision_app/__init__.py

# Expose the core components of the application directly from the main package.
# This creates a clean public API for the vision_app package.
from .camera import CameraManager
from .vision import InferenceEngine
