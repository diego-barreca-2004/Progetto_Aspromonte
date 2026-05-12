# Progetto Aspromonte: 360° Vision Pipeline from an Action Cam to an Open World

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++](https://img.shields.io/badge/C++-13-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU-ee4c2c.svg)](https://pytorch.org/)

## 1. Abstract

The project aims to reproduce, as faithfully as possible, the environment captured by a 360° action camera within a digital twin—specifically, a navigable, video-game style Open World. The input consists of video feeds coupled with GPS metadata, enabling not just a temporal but also a spatial mapping of the observed environment. This automated pipeline is designed to be globally scalable: a 360° video recorded with GPS data in Alaska should be processed just as flawlessly as one recorded in South Africa.

During the development phase, various challenges emerged, primarily geometric. When projecting a continuous spherical environment onto the discrete flat faces of a cube (Cubemap) via *Backward Mapping*, the resulting 2D planes suffer from geometric distortion at their boundaries. Because this pipeline horizontally concatenates these faces into a single, wide panoramic frame for inference, objects spanning across the seams appear bent or spatially disjointed. When standard 2D Convolutional Neural Networks (like YOLO) process this concatenated frame, the convolution kernels interpret these projection artifacts as real physical deformities, leading to blind spots and fragmented bounding boxes. This phenomenon is known as *Seam Discontinuity*.

To address this critical structural flaw, this project explores two divergent paths. The main pipeline establishes a functional baseline using standard 2D object detection on the concatenated cubemap projection, intentionally operating within these inherent boundary limitations. Concurrently, an experimental branch aims to implement state-of-the-art Spherical Convolutional Neural Networks (Spherical CNNs). This advanced path seeks to process the equirectangular format natively on the sphere, entirely bypassing the cubemap projection and mathematically eliminating the seam discontinuity artifacts.

## 2. System Architecture

The core of the project relies on a highly optimized, cross-language data pipeline:

1. **Acquisition:** The 360° video, encoded in ProRes 422 to maximize pixel fidelity (yielding approximately 20GB per minute—making VRAM and memory management a critical component of the architecture), is decoded and split into individual frames using the OpenCV library in C++.
2. **Geometric Engine (C++):** Each frame undergoes a transformation technique known as *Backward Mapping* to convert the equirectangular format into the flat faces of a cubemap. Unlike Forward Mapping, Backward Mapping starts from the destination plane to determine the exact corresponding source pixel on the sphere, avoiding empty pixels. Within `main.cpp`, the mathematical projection (calculating Longitude $\theta$ and Latitude $\phi$) is resolved in $O(1)$ during initialization, generating static Lookup Tables (`map_x`, `map_y`). 
   * **Zero-Copy Memory:** Instead of computationally expensive horizontal concatenations, a single target matrix (`panoramic_frame` of $3072 \times 1024$ pixels) is pre-allocated. The left, front, and right faces are populated directly in place using memory offsets via Regions of Interest (ROI).
   * **Interpolation & Boundary Management:** The matrices are populated using the `remap` function with `INTER_LINEAR` interpolation. Crucially, the `BORDER_WRAP` flag is applied to maintain toroidal continuity when spherical coordinates exceed the $360^\circ$ boundary. The resulting binary stream is piped to standard output.
3. **Inter-Process Communication (IPC):** The Python environment leverages the `subprocess.Popen` module to instantiate the C++ engine as a child process and capture its `stdout` byte stream. This stream is ingested into a 1D `uint8` array via NumPy, reshaped into a 3D tensor ($1024, 3072, 3$), and explicitly duplicated using the `.copy()` method to ensure mutability before reaching PyTorch.
4. **Inference Engine (PyTorch & YOLO):** To benchmark architectural improvements, two distinct versions were developed:
   * **YOLOv5 Pipeline:** Requires manual tensor preprocessing: tensor slicing to convert BGR to RGB, spatial downsampling (to respect the $2^5$ Stride architecture while maintaining the 3:1 aspect ratio), transposition of color channels (HWC to CHW), addition of the batch dimension, and normalization to `[0.0, 1.0]`. A custom Non-Maximum Suppression (NMS) algorithm handles spatial conflicts.
   * **YOLO26 Pipeline:** Encapsulates the entire preprocessing, inference, and NMS pipeline within a unified, highly optimized `.predict()` method, natively exploiting the Tensor Cores.
5. **Post-Processing (Vectorization):** Both models output 2D tensors located in VRAM. To avoid the severe performance penalty of the Python interpreter inside execution loops, the tensors are brought to the CPU and cast to native integers in bulk using NumPy's C-backend vectorization (`.astype(int)`). OpenCV's `.rectangle()` and `.putText()` functions are then invoked to render the detections in real-time.

## 3. Hardware and Software Prerequisites

**Hardware Target**
* Developed and optimized on an MSI Vector 16 HX AI equipped with 32GB of system RAM.
* **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM) for Tensor Core acceleration.

**Software Dependencies**
* NVIDIA CUDA Toolkit and cuDNN.
* **C++ Environment:** `g++-13` (supporting C++20 and C++23 standards), CMake, and OpenCV C++ API.
* **Python Environment:** Python 3.x with a dedicated virtual environment (`venv`).
* **Core Python Libraries:** `torch`, `numpy`, `ultralytics` (for YOLO26), and standard utility scripts for YOLOv5 NMS.

## 4. Build and Installation

The C++ core enforces the **ISO C++23 standard** and utilizes Link-Time Optimization (LTO). It supports hardware-specific binary generation to leverage CPU vectorization instructions (e.g., AVX2).

### C++ Geometric Engine
```bash
# Clone the repository
git clone [https://github.com/diego-barreca-2004/Progetto_Aspromonte.git](https://github.com/diego-barreca-2004/Progetto_Aspromonte.git)
cd Progetto_Aspromonte

# Create the build directory
mkdir build && cd build

# Compile using CMake (with optional flags)
# -DENABLE_GUI=ON (enables OpenCV visual debugging)
# -DENABLE_PROFILING=ON (enables microsecond execution time logging)
# -DUSE_NATIVE_INSTRUCTIONS=ON (enables -march=native for AVX vectorization)
cmake -DUSE_NATIVE_INSTRUCTIONS=ON -DENABLE_PROFILING=ON ..
make

```

### Python Inference Environment

```bash
# Return to the project root
cd ..

# Initialize and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required deep learning dependencies
pip install torch torchvision numpy opencv-python ultralytics

```

## 5. Usage

The architecture is designed to handle the IPC internally within Python, eliminating the need for manual Unix pipes. The Python script autonomously launches the compiled C++ binary and manages the buffer.

To execute the full pipeline (Geometric Engine + Object Detection):

```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# Execute the YOLO26 Engine (auto-loads 'attempt.mov' if no argument is provided)
python3 yolo26.py 

# Or specify a custom video path
python3 yolo26.py /path/to/your/video.mp4

```

## 6. License and Acknowledgments

This project is licensed under the **GNU AGPL-3.0 License**. See the `LICENSE` file for more details.

**Acknowledgments:**

* The core object detection framework relies heavily on the open-source architectures provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
* The experimental branch for resolving *Seam Discontinuity* takes direct inspiration from Google Research's work on [Scalable Spherical CNNs](https://github.com/google-research/spherical-cnn) utilizing the JAX framework.