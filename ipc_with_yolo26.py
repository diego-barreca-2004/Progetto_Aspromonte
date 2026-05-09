import subprocess
import numpy as np
import torch
import cv2
from ultralytics import YOLO

# MODEL INITIALIZATION #

device = torch.device("cuda")
model = YOLO('yolo26s.pt')
model.to(device)

# C++ MANAGEMENT #

process = subprocess.Popen(["./build/Aspromonte"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
width = 1024 * 3 # left, front and right faces of the cubemap
height = 1024
frame_size = width * height * 3

# INFERENCE #

while True:
    raw_bytes = process.stdout.read(frame_size)

    if not raw_bytes: 
        break
    elif len(raw_bytes) < frame_size:
        print("Corrupted frame")

        err = process.stderr.read()
        if err:
            print("C++ error: ", err.decode('utf-8'))
        break

    # NUMPY PIPELINE #

    frame_1d = np.frombuffer(raw_bytes, dtype=np.uint8) 
    frame_3d = np.reshape(frame_1d, (height, width, 3)).copy()
    
    # PYTORCH PIPELINE #

    results = model.predict(source=frame_3d, device=device, verbose=False) # incapsulates tensors slicing, normalizations, transpositions, NMS, ...
    predictions = results[0] # results[0] already contains bounding boxes etc.

    # VRAM -> CPU #

    if len(predictions.boxes) > 0:
        boxes = predictions.boxes.xyxy.cpu().numpy()
        confidences = predictions.boxes.conf.cpu().numpy()
        coco_classes = predictions.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = confidences[i]
            coco_class = int(coco_classes[i])

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
process.stderr.close()
process.wait()
cv2.destroyAllWindows()