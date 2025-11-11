import cv2
import colorsys
import numpy as np
import openvino as ov
import time
import os
import threading
import subprocess as sp
import re
from boxmot import ByteTrack
import dearpygui.dearpygui as dpg

# --- Application Configuration ---
APP_NAME = "Vision App GUI"
MODEL_PATH_XML = "yolo11n_object365.xml"
DEVICE_NAME = "CPU"
CACHE_DIR = "cache"
CAM_WIDTH, CAM_HEIGHT = 1280, 720
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
CLASSES = [
    "Person",
    "Sneakers",
    "Chair",
    "Other Shoes",
    "Hat",
    "Car",
    "Lamp",
    "Glasses",
    "Bottle",
    "Desk",
    "Cup",
    "Street Lights",
    "Cabinet/shelf",
    "Handbag/Satchel",
    "Bracelet",
    "Plate",
    "Picture/Frame",
    "Helmet",
    "Book",
    "Gloves",
    "Storage box",
    "Boat",
    "Leather Shoes",
    "Flower",
    "Bench",
    "Potted Plant",
    "Bowl/Basin",
    "Flag",
    "Pillow",
    "Boots",
    "Vase",
    "Microphone",
    "Necklace",
    "Ring",
    "SUV",
    "Wine Glass",
    "Belt",
    "Monitor/TV",
    "Backpack",
    "Umbrella",
    "Traffic Light",
    "Speaker",
    "Watch",
    "Tie",
    "Trash bin Can",
    "Slippers",
    "Bicycle",
    "Stool",
    "Barrel/bucket",
    "Van",
    "Couch",
    "Sandals",
    "Basket",
    "Drum",
    "Pen/Pencil",
    "Bus",
    "Wild Bird",
    "High Heels",
    "Motorcycle",
    "Guitar",
    "Carpet",
    "Cell Phone",
    "Bread",
    "Camera",
    "Canned",
    "Truck",
    "Traffic cone",
    "Cymbal",
    "Lifesaver",
    "Towel",
    "Stuffed Toy",
    "Candle",
    "Sailboat",
    "Laptop",
    "Awning",
    "Bed",
    "Faucet",
    "Tent",
    "Horse",
    "Mirror",
    "Power outlet",
    "Sink",
    "Apple",
    "Air Conditioner",
    "Knife",
    "Hockey Stick",
    "Paddle",
    "Pickup Truck",
    "Fork",
    "Traffic Sign",
    "Balloon",
    "Tripod",
    "Dog",
    "Spoon",
    "Clock",
    "Pot",
    "Cow",
    "Cake",
    "Dining Table",
    "Sheep",
    "Hanger",
    "Blackboard/Whiteboard",
    "Napkin",
    "Other Fish",
    "Orange/Tangerine",
    "Toiletry",
    "Keyboard",
    "Tomato",
    "Lantern",
    "Machinery Vehicle",
    "Fan",
    "Green Vegetables",
    "Banana",
    "Baseball Glove",
    "Airplane",
    "Mouse",
    "Train",
    "Pumpkin",
    "Soccer",
    "Skiboard",
    "Luggage",
    "Nightstand",
    "Tea pot",
    "Telephone",
    "Trolley",
    "Head Phone",
    "Sports Car",
    "Stop Sign",
    "Dessert",
    "Scooter",
    "Stroller",
    "Crane",
    "Remote",
    "Refrigerator",
    "Oven",
    "Lemon",
    "Duck",
    "Baseball Bat",
    "Surveillance Camera",
    "Cat",
    "Jug",
    "Broccoli",
    "Piano",
    "Pizza",
    "Elephant",
    "Skateboard",
    "Surfboard",
    "Gun",
    "Skating and Skiing shoes",
    "Gas stove",
    "Donut",
    "Bow Tie",
    "Carrot",
    "Toilet",
    "Kite",
    "Strawberry",
    "Other Balls",
    "Shovel",
    "Pepper",
    "Computer Box",
    "Toilet Paper",
    "Cleaning Products",
    "Chopsticks",
    "Microwave",
    "Pigeon",
    "Baseball",
    "Cutting/chopping Board",
    "Coffee Table",
    "Side Table",
    "Scissors",
    "Marker",
    "Pie",
    "Ladder",
    "Snowboard",
    "Cookies",
    "Radiator",
    "Fire Hydrant",
    "Basketball",
    "Zebra",
    "Grape",
    "Giraffe",
    "Potato",
    "Sausage",
    "Tricycle",
    "Violin",
    "Egg",
    "Fire Extinguisher",
    "Candy",
    "Fire Truck",
    "Billiards",
    "Converter",
    "Bathtub",
    "Wheelchair",
    "Golf Club",
    "Briefcase",
    "Cucumber",
    "Cigar/Cigarette",
    "Paint Brush",
    "Pear",
    "Heavy Truck",
    "Hamburger",
    "Extractor",
    "Extension Cord",
    "Tong",
    "Tennis Racket",
    "Folder",
    "American Football",
    "earphone",
    "Mask",
    "Kettle",
    "Tennis",
    "Ship",
    "Swing",
    "Coffee Machine",
    "Slide",
    "Carriage",
    "Onion",
    "Green beans",
    "Projector",
    "Frisbee",
    "Washing Machine/Drying Machine",
    "Chicken",
    "Printer",
    "Watermelon",
    "Saxophone",
    "Tissue",
    "Toothbrush",
    "Ice cream",
    "Hot-air balloon",
    "Cello",
    "French Fries",
    "Scale",
    "Trophy",
    "Cabbage",
    "Hot dog",
    "Blender",
    "Peach",
    "Rice",
    "Wallet/Purse",
    "Volleyball",
    "Deer",
    "Goose",
    "Tape",
    "Tablet",
    "Cosmetics",
    "Trumpet",
    "Pineapple",
    "Golf Ball",
    "Ambulance",
    "Parking meter",
    "Mango",
    "Key",
    "Hurdle",
    "Fishing Rod",
    "Medal",
    "Flute",
    "Brush",
    "Penguin",
    "Megaphone",
    "Corn",
    "Lettuce",
    "Garlic",
    "Swan",
    "Helicopter",
    "Green Onion",
    "Sandwich",
    "Nuts",
    "Speed Limit Sign",
    "Induction Cooker",
    "Broom",
    "Trombone",
    "Plum",
    "Rickshaw",
    "Goldfish",
    "Kiwi fruit",
    "Router/modem",
    "Poker Card",
    "Toaster",
    "Shrimp",
    "Sushi",
    "Cheese",
    "Notepaper",
    "Cherry",
    "Pliers",
    "CD",
    "Pasta",
    "Hammer",
    "Cue",
    "Avocado",
    "Hami melon",
    "Flask",
    "Mushroom",
    "Screwdriver",
    "Soap",
    "Recorder",
    "Bear",
    "Eggplant",
    "Board Eraser",
    "Coconut",
    "Tape Measure/Ruler",
    "Pig",
    "Showerhead",
    "Globe",
    "Chips",
    "Steak",
    "Crosswalk Sign",
    "Stapler",
    "Camel",
    "Formula 1",
    "Pomegranate",
    "Dishwasher",
    "Crab",
    "Hoverboard",
    "Meatball",
    "Rice Cooker",
    "Tuba",
    "Calculator",
    "Papaya",
    "Antelope",
    "Parrot",
    "Seal",
    "Butterfly",
    "Dumbbell",
    "Donkey",
    "Lion",
    "Urinal",
    "Dolphin",
    "Electric Drill",
    "Hair Dryer",
    "Egg tart",
    "Jellyfish",
    "Treadmill",
    "Lighter",
    "Grapefruit",
    "Game board",
    "Mop",
    "Radish",
    "Baozi",
    "Target",
    "French",
    "Spring Rolls",
    "Monkey",
    "Rabbit",
    "Pencil Case",
    "Yak",
    "Red Cabbage",
    "Binoculars",
    "Asparagus",
    "Barbell",
    "Scallop",
    "Noddles",
    "Comb",
    "Dumpling",
    "Oyster",
    "Table Tennis paddle",
    "Cosmetics Brush/Eyeliner Pencil",
    "Chainsaw",
    "Eraser",
    "Lobster",
    "Durian",
    "Okra",
    "Lipstick",
    "Cosmetics Mirror",
    "Curling",
    "Table Tennis",
]


def generate_readable_colors(n):
    """Generate n distinct, saturated but not too bright colors for good visibility."""
    colors = []
    golden_ratio = 0.61803398875
    hue = 0.0

    for i in range(n):
        hue = (hue + golden_ratio) % 1.0
        saturation = 1.0  # maximum color intensity
        value = 0.7  # mildly bright
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(255 * c) for c in rgb))
    return np.array(colors, dtype=np.uint8)


COLORS = generate_readable_colors(len(CLASSES))

# --- Global variables for inter-thread communication ---
latest_frame_lock = threading.Lock()
latest_frame = None
results_lock = threading.Lock()
latest_results = []
stop_event = threading.Event()
camera_change_event = threading.Event()
inference_fps = 0
active_camera_name = None


def find_available_cameras_ffmpeg():
    """Uses FFmpeg to find available cameras on Windows (dshow backend)."""
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        result = sp.run(command, capture_output=True, text=True, check=False)
        output = result.stderr
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in the system PATH.")
        return []

    # Use regex to find camera names listed by FFmpeg
    pattern = re.compile(r'"([^"]+)" \(video\)')
    cameras = pattern.findall(output)
    return cameras


def capture_thread():
    """Captures frames from the selected camera using FFmpeg in a subprocess."""
    global latest_frame, active_camera_name
    proc = None
    current_camera = None

    while not stop_event.is_set():
        # Check if the active camera has changed
        with latest_frame_lock:
            target_camera = active_camera_name

        if target_camera != current_camera:
            # If a capture process is running, stop it
            if proc and proc.poll() is None:
                print(f"Stopping camera: {current_camera}")
                proc.kill()
                proc.wait()

            # Start a new FFmpeg process for the newly selected camera
            if target_camera:
                print(f"Starting camera: {target_camera}")
                command = [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-f",
                    "dshow",
                    "-video_size",
                    f"{CAM_WIDTH}x{CAM_HEIGHT}",
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
                proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL)
                current_camera = target_camera
            else:
                proc = None
                current_camera = None

        # Read frame data from the FFmpeg process stdout
        if proc and proc.poll() is None:
            raw_frame = proc.stdout.read(CAM_WIDTH * CAM_HEIGHT * 3)
            if len(raw_frame) == CAM_WIDTH * CAM_HEIGHT * 3:
                # Convert the raw byte buffer to a NumPy array (image)
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (CAM_HEIGHT, CAM_WIDTH, 3)
                )
                with latest_frame_lock:
                    latest_frame = frame
            else:
                # Handle stream interruption by trying to restart
                print(f"Stream from '{current_camera}' interrupted, restarting...")
                proc.kill()
                proc = None
                current_camera = None
                time.sleep(1)
        else:
            time.sleep(0.1)

    if proc and proc.poll() is None:
        proc.kill()
    print("Camera thread finished.")


def inference_thread(compiled_model):
    """Performs object detection and tracking on frames from the capture thread."""
    global latest_results, inference_fps
    tracker = ByteTrack(verbose=False)
    output_layer = compiled_model.output(0)
    frame_count, start_time = 0, time.time()

    while not stop_event.is_set():
        with latest_frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame_to_process = latest_frame.copy()

        # Pre-process the frame for the model
        frame_height, frame_width = frame_to_process.shape[:2]
        scale = min(INPUT_WIDTH / frame_width, INPUT_HEIGHT / frame_height)
        scaled_w, scaled_h = int(frame_width * scale), int(frame_height * scale)
        resized_frame = cv2.resize(frame_to_process, (scaled_w, scaled_h))
        padded_frame = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        padded_frame[:scaled_h, :scaled_w] = resized_frame
        input_tensor = (
            np.expand_dims(padded_frame, 0).transpose(0, 3, 1, 2).astype(np.float32)
            / 255.0
        )

        # Run inference
        result = compiled_model([input_tensor])[output_layer]
        raw_detections = result.transpose(0, 2, 1)[0]

        # Post-process detections and apply Non-Maximum Suppression (NMS)
        boxes, confidences, class_ids = [], [], []
        for det in raw_detections:
            scores = det[4:]
            conf = np.max(scores)
            if conf > CONFIDENCE_THRESHOLD:
                cls_id = np.argmax(scores)
                cx, cy, w, h = det[:4]
                boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

        # Prepare detections for the tracker
        detections_for_tracker = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections_for_tracker.append(
                    [x, y, x + w, y + h, confidences[i], class_ids[i]]
                )

        # Update the tracker with new detections
        tracked_objects = tracker.update(
            np.array(detections_for_tracker), frame_to_process
        )

        # Store final results, scaling coordinates back to original frame size
        current_results = []
        if len(tracked_objects) > 0:
            for x1, y1, x2, y2, track_id, conf, class_id, _ in tracked_objects:
                x1_s, y1_s, x2_s, y2_s = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale),
                )
                current_results.append(
                    [(x1_s, y1_s, x2_s, y2_s), int(class_id), conf, int(track_id)]
                )

        with results_lock:
            latest_results = current_results

        # Calculate inference FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            inference_fps = frame_count / elapsed
            frame_count, start_time = 0, time.time()


def main():
    global active_camera_name

    # Find cameras and initialize OpenVINO model
    available_cameras = find_available_cameras_ffmpeg()
    if not available_cameras:
        print("No cameras found!")
        return
    active_camera_name = available_cameras[0]

    core = ov.Core()
    model = core.read_model(MODEL_PATH_XML)
    compiled_model = core.compile_model(model, DEVICE_NAME, {"CACHE_DIR": CACHE_DIR})

    # Start capture and inference threads
    capture_t = threading.Thread(target=capture_thread, daemon=True)
    inference_t = threading.Thread(
        target=inference_thread, args=(compiled_model,), daemon=True
    )
    capture_t.start()
    camera_change_event.set()
    inference_t.start()

    # --- Dear PyGui Setup ---
    dpg.create_context()

    # Create a texture to display the video frames
    texture_data = np.zeros((CAM_HEIGHT, CAM_WIDTH, 4), dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(
            CAM_WIDTH,
            CAM_HEIGHT,
            texture_data,
            format=dpg.mvFormat_Float_rgba,
            tag="video_texture",
        )

    def switch_camera_callback(sender, app_data):
        """Callback to change the active camera."""
        global active_camera_name
        with latest_frame_lock:
            active_camera_name = app_data
        camera_change_event.set()

    # Create the main window and UI elements
    with dpg.window(label="Vision App", tag="primary_window"):
        with dpg.group(horizontal=True):
            dpg.add_text("Display FPS: 0.00", tag="display_fps_text")
            dpg.add_spacer(width=50)
            dpg.add_text("Inference FPS: 0.00", tag="inference_fps_text")

        if len(available_cameras) > 1:
            dpg.add_combo(
                items=available_cameras,
                label="Select Camera",
                default_value=active_camera_name,
                callback=switch_camera_callback,
            )

        # A drawlist will hold the video frame and bounding box overlays
        with dpg.drawlist(width=CAM_WIDTH, height=CAM_HEIGHT, tag="video_drawlist"):
            dpg.draw_image(
                "video_texture",
                pmin=(0, 0),
                pmax=(CAM_WIDTH, CAM_HEIGHT),
                tag="video_draw_image",
            )
            # A separate layer for drawings ensures they are on top of the image
            dpg.add_draw_layer(tag="results_layer", parent="video_drawlist")

    dpg.create_viewport(
        title=APP_NAME,
        width=CAM_WIDTH + 40,
        height=CAM_HEIGHT + 150,
        always_on_top=True,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    # Focus the primary window
    dpg.focus_item("primary_window")

    # --- Main GUI Loop ---
    while dpg.is_dearpygui_running():
        # Update the video texture with the latest frame
        with latest_frame_lock:
            if latest_frame is not None:
                rgba_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGBA)
                dpg.set_value("video_texture", (rgba_frame.astype(np.float32) / 255.0))

        # Clear previous frame's drawings
        dpg.delete_item("results_layer", children_only=True)

        with results_lock:
            results_to_draw = latest_results

        # Draw new bounding boxes and labels for the current frame
        for box, cls_id, conf, track_id in results_to_draw:
            if cls_id < len(CLASSES):
                x1, y1, x2, y2 = map(int, box)

                color = tuple(COLORS[cls_id]) + (255,)
                label = f"ID:{track_id} {CLASSES[cls_id]}: {conf:.2f}"

                dpg.draw_rectangle(
                    (x1, y1), (x2, y2), color=color, thickness=3, parent="results_layer"
                )

                # Position the text label above the box
                label_height = 20
                padding = 5
                text_y = y1 - label_height - padding
                if text_y < 0:
                    text_y = y1 + padding  # place below box if not enough space above

                # Make background semi-transparent for better readability
                text_bg_color = (255, 255, 255, 180)
                text_color = (0, 0, 0, 255)

                # Estimate background width based on label length and font size
                approx_char_width = label_height * 0.55
                bg_width = int(len(label) * approx_char_width)
                bg_height = label_height + 4

                bg_x2 = min(x1 + bg_width + 6, CAM_WIDTH)
                bg_y2 = min(text_y + bg_height, CAM_HEIGHT)

                # Box thickness
                box_thickness = 3

                # Draw semi-transparent rectangle background
                dpg.draw_rectangle(
                    (max(0, x1 + box_thickness), max(0, text_y - 2)),
                    (bg_x2, bg_y2),
                    color=text_bg_color,
                    fill=text_bg_color,
                    parent="results_layer",
                )
                # Actual text
                dpg.draw_text(
                    (x1 + box_thickness + 2, text_y),
                    label,
                    color=text_color,
                    size=label_height,
                    parent="results_layer",
                )

        # Update FPS counters
        dpg.set_value("display_fps_text", f"Display FPS: {dpg.get_frame_rate():.2f}")
        dpg.set_value("inference_fps_text", f"Inference FPS: {inference_fps:.2f}")

        dpg.render_dearpygui_frame()

    # --- Cleanup ---
    print("Closing application...")
    stop_event.set()
    capture_t.join()
    inference_t.join()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
