import cv2
import numpy as np
import openvino as ov
import time
import os
import threading
from boxmot import ByteTrack

# --- App Configuration ---
# General application settings
APP_NAME = "Vision App - YOLOv11n + ByteTrack + OpenVINO"
MODEL_PATH_XML = "yolo11n_object365.xml"  # Path to the OpenVINO IR model
DEVICE_NAME = "CPU"  # Device to run inference on (e.g., "CPU", "GPU")
CACHE_DIR = "cache"  # Directory for OpenVINO model caching to speed up startup

# --- Camera Configuration ---
# Settings for the webcam capture
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# --- Model Configuration ---
# Parameters for the YOLO model and post-processing
INPUT_WIDTH = 640  # The width the model expects
INPUT_HEIGHT = 640  # The height the model expects
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to consider a detection
IOU_THRESHOLD = 0.5  # Threshold for Non-Maximum Suppression (NMS)

# Full list of 365 classes from the Objects365 dataset
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
    "Orange/T Tangerine",
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
# Generate a unique color for each class for visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# --- Global synchronized variables ---
# These variables are shared between threads and require locks for safe access
latest_frame_lock = threading.Lock()
latest_frame = None  # Stores the most recent frame from the camera
results_lock = threading.Lock()
latest_results = []  # Stores the most recent tracking results
stop_event = threading.Event()  # A signal for all threads to gracefully exit
inference_fps = 0  # FPS calculated in the inference thread


# --- Threads ---


def capture_thread(cap):
    """
    This thread's only job is to continuously read frames from the camera.
    It runs in the background to ensure the main thread always has the latest frame
    without being blocked by the camera's I/O.
    """
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # Use a lock to safely write the new frame to the global variable
        with latest_frame_lock:
            latest_frame = frame


def inference_thread(compiled_model):
    """
    This thread handles all the heavy lifting: pre-processing, inference,
    post-processing (NMS), and object tracking. It runs independently to
    avoid slowing down the camera capture or the display.
    """
    global latest_results, inference_fps

    # Initialize the ByteTrack tracker
    tracker = ByteTrack(per_class=True, nr_classes=len(CLASSES), verbose=False)

    # Get the model's output layer
    output_layer = compiled_model.output(0)

    frame_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        # Safely read the latest frame for processing
        with latest_frame_lock:
            if latest_frame is None:
                time.sleep(0.01)  # Wait if no frame is available yet
                continue
            frame_to_process = latest_frame.copy()

        # --- Pre-processing ---
        # Resize and pad the image to match the model's input size (letterboxing)
        frame_height, frame_width = frame_to_process.shape[:2]
        scale = min(INPUT_WIDTH / frame_width, INPUT_HEIGHT / frame_height)
        scaled_w, scaled_h = int(frame_width * scale), int(frame_height * scale)
        resized_frame = cv2.resize(frame_to_process, (scaled_w, scaled_h))

        # Create a padded frame and place the resized frame in it
        padded_frame = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
        padded_frame[:scaled_h, :scaled_w] = resized_frame

        # Prepare the tensor for the model: (H,W,C) -> (1,C,H,W) and normalize
        input_tensor = (
            np.expand_dims(padded_frame, 0).transpose(0, 3, 1, 2).astype(np.float32)
            / 255.0
        )

        # --- Inference ---
        # Run the model on the pre-processed frame
        result = compiled_model([input_tensor])[output_layer]

        # --- Post-processing ---
        # Transpose the output to iterate through detections easily
        raw_detections = result.transpose(0, 2, 1)[0]
        boxes, confidences, class_ids = [], [], []

        # Parse the raw detections
        for det in raw_detections:
            scores = det[4:]
            conf = np.max(scores)
            if conf > CONFIDENCE_THRESHOLD:
                cls_id = np.argmax(scores)
                # Convert YOLO's (center_x, center_y, width, height) to (x, y, w, h) for NMS
                cx, cy, w, h = det[:4]
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                boxes.append([x1, y1, int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(cls_id)

        # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

        # Prepare detections in the format required by the tracker
        detections_for_tracker = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # boxmot expects (x1, y1, x2, y2, conf, cls)
                detections_for_tracker.append(
                    [x, y, x + w, y + h, confidences[i], class_ids[i]]
                )

        # --- Tracking ---
        # Update the tracker with the new detections
        tracked_objects = tracker.update(
            np.array(detections_for_tracker), frame_to_process
        )

        # --- Format and Scale Results ---
        # Scale the bounding box coordinates back to the original frame size
        current_results = []
        if len(tracked_objects) > 0:
            for x1, y1, x2, y2, track_id, conf, class_id, _ in tracked_objects:
                x1_s = int(x1 / scale)
                y1_s = int(y1 / scale)
                x2_s = int(x2 / scale)
                y2_s = int(y2 / scale)
                current_results.append(
                    [(x1_s, y1_s, x2_s, y2_s), int(class_id), conf, int(track_id)]
                )

        # Safely update the shared results variable
        with results_lock:
            latest_results = current_results

        # Calculate inference FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            inference_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()


def main():
    """
    Main function to initialize and run the application.
    It sets up OpenVINO, the camera, starts the threads, and handles the display loop.
    """
    # --- Initialization ---
    # Initialize OpenVINO Core and compile the model for the target device
    core = ov.Core()
    model = core.read_model(MODEL_PATH_XML)
    compiled_model = core.compile_model(model, DEVICE_NAME, {"CACHE_DIR": CACHE_DIR})

    # Initialize video capture with the MSMF backend for better performance on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Configure camera properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set minimal buffer size for low latency

    # Print camera settings for verification
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera configured: {int(actual_width)}x{int(actual_height)} @ {actual_fps:.2f} FPS"
    )

    # --- Start Threads ---
    # Create and start the capture and inference threads
    capture_t = threading.Thread(target=capture_thread, args=(cap,))
    inference_t = threading.Thread(target=inference_thread, args=(compiled_model,))
    capture_t.start()
    inference_t.start()

    # --- Main Display Loop ---
    display_fps, frame_count, start_time = 0, 0, time.time()
    TARGET_FPS = 30
    FRAME_TIME = 1 / TARGET_FPS  # Time per frame to achieve target FPS

    while not stop_event.is_set():
        loop_start_time = time.time()

        # Safely get the latest frame and results for display
        with latest_frame_lock:
            if latest_frame is None:
                time.sleep(FRAME_TIME)
                continue
            display_frame = latest_frame.copy()

        with results_lock:
            results_to_draw = latest_results

        # Draw the bounding boxes and labels on the frame
        for box, cls_id, conf, track_id in results_to_draw:
            x1, y1, x2, y2 = box
            color = [int(c) for c in COLORS[cls_id]]
            label = f"ID:{track_id} {CLASSES[cls_id]}: {conf:.2f}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Calculate and display FPS metrics
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        cv2.putText(
            display_frame,
            f"Display FPS: {display_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Inference FPS: {inference_fps:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Show the final frame
        cv2.imshow(APP_NAME, display_frame)

        # Regulate the loop to approximate the TARGET_FPS
        elapsed_loop = time.time() - loop_start_time
        sleep_time = FRAME_TIME - elapsed_loop
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Check for exit condition (user presses 'q' or closes the window)
        if (
            cv2.waitKey(1) & 0xFF == ord("q")
            or cv2.getWindowProperty(APP_NAME, cv2.WND_PROP_VISIBLE) < 1
        ):
            stop_event.set()
            break

    # --- Cleanup ---
    # Wait for threads to finish before exiting
    print("Stopping threads...")
    capture_t.join()
    inference_t.join()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
