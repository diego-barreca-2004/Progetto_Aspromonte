import sys
import os
import subprocess
import numpy as np
import torch
import cv2
import warnings

# Suppress unavoidable but harmless FutureWarnings (e.g., from torch.hub)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------ #
#  MODEL INITIALIZATION
# ------------------------------------------------------------------ #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre‑trained YOLOv5s model from the Ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)      # move model to GPU if available
model.eval()           # set to evaluation mode (disables dropout / BN updates)

# YOLOv5 requires explicit Non‑Maximum Suppression, which YOLO26 encapsulates internally
from utils.general import non_max_suppression

# ------------------------------------------------------------------ #
#  VIDEO INPUT
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

# Spawn the C++ process that converts equirectangular → cubemap and streams raw pixels
process = subprocess.Popen([EXECUTABLE_PATH, VIDEO_PATH], stdout=subprocess.PIPE)

WIDTH = 1024 * 3                   # Cubemap 3x1 faces
HEIGHT = 1024
COLOR_CHANNELS = 3
frame_size = WIDTH * HEIGHT * COLOR_CHANNELS

# ------------------------------------------------------------------ #
#  MAIN INFERENCE LOOP
# ------------------------------------------------------------------ #

while True:
    # -- Read raw frame from C++ process ----------------------------------
    raw_bytes = process.stdout.read(frame_size)
    if not raw_bytes:
        break                          # end of stream
    if len(raw_bytes) < frame_size:
        print("WARNING: Incomplete frame received. Exiting stream...")
        break

    # -- NumPy preprocessing pipeline ------------------------------------
    # Decode the flat byte buffer into a 3‑channel cubemap image
    frame_1d = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame_3d = np.reshape(frame_1d, (HEIGHT, WIDTH, COLOR_CHANNELS)).copy()

    # Spatial downsampling to reduce VRAM consumption.
    # The network width and height must be multiples of 32 (YOLOv5 stride constraint)
    # while preserving the 3:1 cubemap aspect ratio.
    NET_W, NET_H = 960, 320
    scale_x = WIDTH / NET_W
    scale_y = HEIGHT / NET_H
    frame_resized_net = cv2.resize(frame_3d, (NET_W, NET_H))

    # Convert BGR (OpenCV) → RGB (PyTorch), reorder to CHW, add batch dim, normalise
    frame_rgb = frame_resized_net[:, :, ::-1]               # BGR → RGB
    frame_chw = np.transpose(frame_rgb, (2, 0, 1))          # HWC → CHW
    frame_batch = np.expand_dims(frame_chw, 0)              # (C, H, W) → (1, C, H, W)
    frame_norm = frame_batch.astype(np.float32) / 255.0     # normalise to [0, 1]

    # -- PyTorch inference ------------------------------------------------
    frame_tensor = torch.from_numpy(frame_norm).to(device)

    with torch.no_grad():
        output = model(frame_tensor)
        # YOLOv5 may return a tuple; take the first element (raw predictions)
        raw_predictions = output[0] if isinstance(output, tuple) else output
        # Apply Non‑Maximum Suppression (conf ≥ 0.25, IoU ≥ 0.45)
        predictions = non_max_suppression(raw_predictions.clone(),
                                          conf_thres=0.25, iou_thres=0.45)[0]

    # -- Post‑processing & drawing ----------------------------------------
    # predictions shape: [N, 6] → (x1, y1, x2, y2, confidence, class)
    if predictions is not None and len(predictions) > 0:
        # Move tensors to CPU and scale coordinates back to original cubemap size
        boxes = predictions[:, :4].cpu().numpy()
        scales = np.array([scale_x, scale_y, scale_x, scale_y])
        boxes = (boxes * scales).astype(int)

        confidences = predictions[:, 4].cpu().numpy()
        coco_classes = predictions[:, 5].cpu().numpy().astype(int)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            coco_class = coco_classes[i]

            class_name = model.names[coco_class]
            label = f"{class_name} {int(confidence * 100)}%"

            # Draw bounding box and label on the original frame
            cv2.rectangle(frame_3d, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_3d, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # -- Display (resized to fit screen) ----------------------------------
    frame_resized = cv2.resize(frame_3d, (2560, 1600))
    cv2.imshow("Progetto Aspromonte - Legacy YOLOv5", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------------ #
#  CLEANUP
# ------------------------------------------------------------------ #

process.stdout.close()
process.wait()
cv2.destroyAllWindows()