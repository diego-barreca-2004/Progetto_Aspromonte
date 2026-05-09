import subprocess
import numpy as np
import torch
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # to avoid unavoidable harmless warnings in the terminal

# MODEL INITIALIZATION #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # to use NVIDIA GPU if available
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # to use pre-trained YOLOv5s
model.to(device) # to exploit VRAM
model.eval() # disables training
from utils.general import non_max_suppression # (run time dependency) to use the Non Max Suppression function, which is instead incapsulated in YOLO26

# C++ MANAGEMENT #

process = subprocess.Popen(["./build/Aspromonte"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # opens a pipe with the c++ executable
width = 1024 * 3 # left, front and right faces of the cubemap
height = 1024
frame_size = width * height * 3

# INFERENCE #

while True: # infinite loop
    raw_bytes = process.stdout.read(frame_size)
    if not raw_bytes : # exits when .read() returns b'', which happens at the end of the processed video
        break
    elif len(raw_bytes) < frame_size:
        print("Corrupted frame")

        err = process.stderr.read()
        if err:
            print("C++ error: ", err.decode('utf-8')) # debugging
        break

    # NUMPY PIPELINE #

    frame_1d = np.frombuffer(raw_bytes, dtype=np.uint8) # monodimensional array
    frame_3d = np.reshape(frame_1d, (height, width, 3)).copy() # 1D -> 3D array... and .copy() needed: without it, it would be a read-only array
    frame_rgb = frame_3d[:, :, ::-1] # BGR -> RGB colors (tensor slicing used to invert only the order of the color channels)
    frame_chw = np.transpose(frame_rgb, (2, 0, 1)) # puts the color channels first to optimize spatial locality for neural network convolutions
    frame_batch = np.expand_dims(frame_chw, 0) # adds the batch index
    frame_norm = frame_batch.astype(np.float32) / 255.0 # normalizes to [0.0, 1.0]

    # PYTORCH PIPELINE #

    frame_tensor = torch.from_numpy(frame_norm).to(device) # converts the NumPy tensor in a PyTorch tensor and then allocates it in the VRAM 
    with torch.no_grad(): # disables the gradients tracking
        output = model(frame_tensor) # performs inference
        raw_predictions = output[0] if isinstance(output, tuple) else output # predictions at index [0]... if it's not a tuple then uses the whole tensor
        predictions = non_max_suppression(raw_predictions.clone(), conf_thres=0.25, iou_thres=0.45)[0] # returns a tensor list for each image in the batch and removes boxes with low confidence and duplicates... and .clone() needed to have a writable copy in VRAM

    # VRAM -> CPU #

    # The returned tensor is bidimensional: [N, 6] -> (x1, y1, x2, y2, confidence, class)
    if predictions is not None and len(predictions) > 0: # if there's at least an object
        boxes = predictions[:, :4].cpu().numpy() # (x1, y1) are the coordinates of the upper left corner and (x2, y2) are the coordinates of the bottom right corner
        confidences = predictions[:, 4].cpu().numpy() # confidence between 0.0 and 1.0
        coco_classes = predictions[:, 5].cpu().numpy() # COCO class ID, although for example a tree class is missing (time for the 5070 Ti!!!)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i]) # removes decimal figures from these new pixel coordinates and passes the corresponding value to each of the variables
            confidence = confidences[i]
            coco_class = int(coco_classes[i]) # removes decimal figures from the COCO class IDs

            class_name = model.names[coco_class] # gets the COCO class name
            label = f"{class_name} {int(confidence * 100)}%" # formatted string literal

            cv2.rectangle(frame_3d, (x1, y1), (x2, y2), (0, 255, 0), 2) # creates a green rectangle for each of the bounding boxes with some thickness (2)
            cv2.putText(frame_3d, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # adds the formatted green bold text

    # OUTPUT ON SCREEN #

    frame_resized = cv2.resize(frame_3d, (2560, 1600)) # resizes to fit my screen
    cv2.imshow("Progetto Aspromonte first attempt with YOLO5", frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # waits each ms for a user input ('q'). 0xFF is needed to get each time the real 'q' ASCII value (113)
        break

process.stdout.close()
process.stderr.close()
process.wait()
cv2.destroyAllWindows() # closes the OpenCV window