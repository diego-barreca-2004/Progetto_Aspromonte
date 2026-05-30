import sys
import os
import subprocess
import numpy as np
import torch
import cv2
import math
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# ------------------------------------------------------------------ #
#  CONFIGURATION
# ------------------------------------------------------------------ #

USE_GUI = True                     # Set to False for headless mode
MODEL_WEIGHTS = 'yolo26s-seg.pt'   # YOLO model weights (replace with your fine-tuned model)
SSIM_THRESHOLD = 0.92              # Structural similarity threshold (1.0 = identical). Lower -> fewer keyframes
MIN_FRAME_GAP = 30                 # Minimum number of frames between consecutive keyframes
SAVE_ENHANCED = True               # Save CLAHE-enhanced frames (recommended for training)
SHOW_ONLY_KEYFRAMES = False        # If True, display only keyframes; if False, smooth continuous video

# ------------------------------------------------------------------ #
#  FUNCTIONS
# ------------------------------------------------------------------ #

def get_unreal_engine_vector(obj_x_root, obj_y_root, FACE_DIM=1024):
    """
    Compute the 3D direction vector in Unreal Engine's left-handed Z-up coordinate system.
    Returns (X_ue, Y_ue, Z_ue), azimuth, elevation.
    """
    cubemap_face_index = int(obj_x_root // FACE_DIM)
    face_x_root = obj_x_root % FACE_DIM

    u = (face_x_root / float(FACE_DIM)) * 2.0 - 1.0
    v = (obj_y_root / float(FACE_DIM)) * 2.0 - 1.0

    if cubemap_face_index == 0:      # Left face
        X_ue, Y_ue = u, -1.0
    elif cubemap_face_index == 1:    # Front face
        X_ue, Y_ue = 1.0, u
    else:                            # Right face
        X_ue, Y_ue = -u, 1.0

    Z_ue = -v

    azimuth = math.degrees(math.atan2(Y_ue, X_ue))
    elevation = math.degrees(math.asin(Z_ue / math.sqrt(X_ue**2 + Y_ue**2 + Z_ue**2)))

    return (X_ue, Y_ue, Z_ue), azimuth, elevation


def enhance_low_light(image_bgr, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L channel
    of the LAB colour space to brighten shadows and improve detection in low-light
    environments (e.g., dense forest undergrowth).
    Returns the enhanced BGR image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------ #
#  MODEL INITIALIZATION
# ------------------------------------------------------------------ #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_WEIGHTS)
model.to(device)

# ------------------------------------------------------------------ #
#  DATASET SETUP
# ------------------------------------------------------------------ #

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
frame_counter = 0

# ------------------------------------------------------------------ #
#  INPUT
# ------------------------------------------------------------------ #

DEFAULT_VIDEO_PATH = "attempt.mov"
EXECUTABLE_PATH = "./build/Aspromonte"

if len(sys.argv) >= 2:
    VIDEO_PATH = sys.argv[1]
elif os.path.exists(DEFAULT_VIDEO_PATH):
    VIDEO_PATH = DEFAULT_VIDEO_PATH
    print(f"\nAuto-loading local default video: '{VIDEO_PATH}'\n")
else:
    print(f"Usage: python3 {sys.argv[0]} <video_path>")
    sys.exit(1)

# Spawn the C++ process that streams raw cubemap bytes
process = subprocess.Popen([EXECUTABLE_PATH, VIDEO_PATH], stdout=subprocess.PIPE)

WIDTH = 1024 * 3                   # Cubemap 3x1 faces
HEIGHT = 1024
COLOR_CHANNELS = 3
frame_size = WIDTH * HEIGHT * COLOR_CHANNELS

# ------------------------------------------------------------------ #
#  KEYFRAME SELECTION STATE
# ------------------------------------------------------------------ #

last_gray_frame = None             # Grayscale, downscaled reference of the last keyframe
frames_since_last_keyframe = 0

# ------------------------------------------------------------------ #
#  MAIN PROCESSING LOOP
# ------------------------------------------------------------------ #

while True:
    # -- Read raw frame from C++ process ----------------------------------
    raw_bytes = process.stdout.read(frame_size)
    if not raw_bytes:
        break
    if len(raw_bytes) < frame_size:
        print("WARNING: Incomplete frame received. Exiting stream...")
        break

    # Decode the cubemap image (3x1 faces) into a 3D NumPy array
    frame_1d = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame_3d = np.reshape(frame_1d, (HEIGHT, WIDTH, COLOR_CHANNELS)).copy()

    # -- Keyframe detection using SSIM ------------------------------------
    # Downscale and blur to suppress camera noise and subtle leaf movement
    small = cv2.resize(frame_3d, (256, 256))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.GaussianBlur(gray_small, (5, 5), 0)

    is_keyframe = True

    if last_gray_frame is not None:
        # Compute structural similarity (1.0 = identical)
        score = ssim(gray_small, last_gray_frame, data_range=255)
        if score >= SSIM_THRESHOLD:
            is_keyframe = False

    # Enforce a minimum temporal gap between consecutive keyframes
    if is_keyframe and frames_since_last_keyframe < MIN_FRAME_GAP:
        is_keyframe = False

    # Update reference and counter
    if is_keyframe:
        last_gray_frame = gray_small.copy()
        frames_since_last_keyframe = 0
    else:
        frames_since_last_keyframe += 1

    # -- Prepare display frame (always use the current raw frame) ---------
    display_frame = frame_3d.copy()

    # -- Keyframe processing: dataset saving & inference ------------------
    if is_keyframe:
        # Enhance low‑light areas with CLAHE (recommended for training)
        if SAVE_ENHANCED:
            frame_to_save = enhance_low_light(frame_3d)
        else:
            frame_to_save = frame_3d.copy()

        # Save the clean (annotation‑free) image to the dataset
        export_path = os.path.join(DATASET_DIR, f"frame_{frame_counter:04d}.jpg")
        cv2.imwrite(export_path, frame_to_save)
        frame_counter += 1

        # Run YOLO inference on the enhanced frame
        results = model.predict(source=frame_to_save, device=device, verbose=False)
        predictions = results[0]

        # Draw bounding boxes on the display frame if detections exist
        if USE_GUI and len(predictions.boxes) > 0:
            # Move tensors to CPU and convert to NumPy arrays
            boxes = predictions.boxes.xyxy.cpu().numpy().astype(int)
            confidences = predictions.boxes.conf.cpu().numpy()
            coco_classes = predictions.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = confidences[i]
                coco_class = coco_classes[i]
                class_name = model.names[coco_class]

                # Root point – bottom centre of the bounding box (ground contact)
                obj_x_root = int((x1 + x2) / 2)
                obj_y_root = int(y2)

                # Compute 3D direction vector for Unreal Engine placement
                ue_vector, azimuth, elevation = get_unreal_engine_vector(obj_x_root, obj_y_root)

                # Draw bounding box and root point
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display_frame, (obj_x_root, obj_y_root), 8, (0, 0, 255), -1)

                # Labels: class name + confidence, and spherical coordinates
                label = f"{class_name} {int(confidence * 100)}%"
                coord_label = f"Az: {azimuth:.1f} | El: {elevation:.1f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, coord_label, (obj_x_root + 10, obj_y_root),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # -- Display output (smooth video or only keyframes) -------------------
    if USE_GUI:
        if SHOW_ONLY_KEYFRAMES and not is_keyframe:
            # In keyframe‑only mode, skip non‑keyframes to avoid visual clutter
            pass
        else:
            cv2.imshow("Progetto Aspromonte", cv2.resize(display_frame, (2560, 1600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# ------------------------------------------------------------------ #
#  CLEANUP
# ------------------------------------------------------------------ #

process.stdout.close()
process.wait()
cv2.destroyAllWindows()