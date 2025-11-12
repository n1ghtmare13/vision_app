# /vision_app/vision_app/ui_dpg/main_window.py
# Main window and display loop for the Dear PyGui interface.

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from . import drawing
from .. import config


class AppGui:
    def __init__(self, camera_manager, inference_engine, available_cameras):
        self.camera = camera_manager
        self.engine = inference_engine
        self.available_cameras = available_cameras

    def _switch_camera_callback(self, sender, app_data):
        """Callback to change the active camera."""
        self.camera.set_camera(app_data)

    def run(self):
        """Sets up and runs the Dear PyGui main loop."""
        dpg.create_context()

        # Create a texture to display the video frames
        texture_data = np.zeros(
            (config.CAM_HEIGHT, config.CAM_WIDTH, 4), dtype=np.float32
        )
        with dpg.texture_registry():
            dpg.add_raw_texture(
                config.CAM_WIDTH,
                config.CAM_HEIGHT,
                texture_data,
                format=dpg.mvFormat_Float_rgba,
                tag="video_texture",
            )

        # Create the main window and UI elements
        with dpg.window(label="Vision App", tag="primary_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("Display FPS: 0.00", tag="display_fps_text")
                dpg.add_spacer(width=50)
                dpg.add_text("Inference FPS: 0.00", tag="inference_fps_text")

            if len(self.available_cameras) > 1:
                dpg.add_combo(
                    items=self.available_cameras,
                    label="Select Camera",
                    default_value=self.camera.active_camera,
                    callback=self._switch_camera_callback,
                )

            with dpg.drawlist(
                width=config.CAM_WIDTH, height=config.CAM_HEIGHT, tag="video_drawlist"
            ):
                dpg.draw_image(
                    "video_texture",
                    pmin=(0, 0),
                    pmax=(config.CAM_WIDTH, config.CAM_HEIGHT),
                )
                dpg.add_draw_layer(tag="results_layer", parent="video_drawlist")

        dpg.create_viewport(
            title=config.APP_NAME_GUI,
            width=config.CAM_WIDTH + 40,
            height=config.CAM_HEIGHT + 150,
            always_on_top=True,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        dpg.focus_item("primary_window")

        # Main GUI Loop
        while dpg.is_dearpygui_running():
            frame = self.camera.get_frame()
            if frame is not None:
                rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                dpg.set_value("video_texture", (rgba_frame.astype(np.float32) / 255.0))

            dpg.delete_item("results_layer", children_only=True)
            results = self.engine.get_results()
            if results:
                drawing.draw_results(results, parent_layer="results_layer")

            dpg.set_value(
                "display_fps_text", f"Display FPS: {dpg.get_frame_rate():.2f}"
            )
            dpg.set_value("inference_fps_text", f"Inference FPS: {self.engine.fps:.2f}")

            dpg.render_dearpygui_frame()

        dpg.destroy_context()
