import sys
import os
import subprocess
import numpy as np
import torch
import cv2
from ultralytics import YOLO

# MODEL INITIALIZATION #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolo26s.pt')
model.to(device)

# C++ MANAGEMENT #

DEFAULT_VIDEO_PATH = "attempt.mov" # local video input
EXECUTABLE_PATH = "./build/Aspromonte"

if len(sys.argv) >= 2: # to use the video path passed via command line
    VIDEO_PATH = sys.argv[1]
elif os.path.exists(DEFAULT_VIDEO_PATH): # to use the local video input
    VIDEO_PATH = DEFAULT_VIDEO_PATH
    print(f"\nAuto-loading local default video: '{VIDEO_PATH}'\n")
else: # to force to pass the video path via command line
    print(f"Correct usage is: python3 {sys.argv[0]} <video_path>")
    sys.exit(1)

process = subprocess.Popen([EXECUTABLE_PATH, VIDEO_PATH], stdout=subprocess.PIPE)

WIDTH = 1024 * 3
HEIGHT = 1024
COLOR_CHANNELS = 3
frame_size = WIDTH * HEIGHT * COLOR_CHANNELS

# INFERENCE #

while True:
    raw_bytes = process.stdout.read(frame_size)

    if not raw_bytes: 
        break
    elif len(raw_bytes) < frame_size:
        print("WARNING! Incomplete frame received or End of File reached. Exiting stream...")
        break

    # NUMPY PIPELINE #

    frame_1d = np.frombuffer(raw_bytes, dtype=np.uint8) 
    frame_3d = np.reshape(frame_1d, (HEIGHT, WIDTH, COLOR_CHANNELS)).copy()
    
    # PYTORCH PIPELINE #

    results = model.predict(source=frame_3d, device=device, verbose=False) # incapsulates tensors slicing, normalizations, transpositions, NMS, ...
    predictions = results[0] # results[0] already contains bounding boxes etc.

    # VRAM -> CPU #

    if len(predictions.boxes) > 0:
        boxes = predictions.boxes.xyxy.cpu().numpy().astype(int)
        confidences = predictions.boxes.conf.cpu().numpy()
        coco_classes = predictions.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            coco_class = coco_classes[i]

            class_name = model.names[coco_class]
            label = f"{class_name} {int(confidence * 100)}%"

            cv2.rectangle(frame_3d, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_3d, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # OUTPUT ON SCREEN #

    frame_resized = cv2.resize(frame_3d, (2560, 1600))
    cv2.imshow("Progetto Aspromonte first attempt with YOLO26", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.stdout.close()
process.wait()
cv2.destroyAllWindows()