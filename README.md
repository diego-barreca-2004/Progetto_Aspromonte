# Progetto Aspromonte: 360° Vision Pipeline from an Action Cam to an Open World

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++](https://img.shields.io/badge/C++-13-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU-ee4c2c.svg)](https://pytorch.org/)

## 1. Abstract

The project aims to reproduce, as faithfully as possible, the environment captured by a 360° action camera within a digital twin—specifically, a navigable, video-game style Open World. The input consists of video feeds coupled with GPS metadata, enabling not just a temporal but also a spatial mapping of the observed environment. This automated pipeline is designed to be globally scalable: a 360° video recorded with GPS data in Alaska should be processed just as flawlessly as one recorded in South Africa.

During the development phase, two major structural challenges emerged:
1. **Seam Discontinuity:** When projecting a continuous spherical environment onto the discrete flat faces of a cube (Cubemap), objects spanning across the seams appear bent or spatially disjointed. Convolutional Neural Networks interpret these projection artifacts as physical deformities.
2. **Root-Point Anchoring:** Traditional bounding box regressions (like YOLO) inherently fail to provide the surgical 3D anchoring required for spawning Digital Twin assets. A bounding box center often floats in empty space, whereas spawning a 3D object (e.g., a tree or a building) in an Open World engine requires its exact ground-contact footprint.

To address these limitations, the project is structured into two parallel branches:
* **The Main Branch (Real-Time Baseline):** Establishes a highly optimized, high-FPS baseline using standard 2D object detection (YOLO) on the concatenated cubemap projection, intentionally operating within these inherent boundary and bounding-box limitations to prioritize real-time processing.
* **The Experimental Branch (High-Fidelity Topography):** Abandons bounding boxes and planar projections entirely. It utilizes the HEALPix (Hierarchical Equal Area isoLatitude Pixelization) grid and Spherical Transformers to perform precise **Spherical Semantic Segmentation**. By classifying every single pixel natively on the sphere, this branch extracts exact object footprints to achieve mathematically perfect 3D asset instantiation.

## 2. System Architecture

The core of the project relies on a highly optimized, cross-language data pipeline:

1. **Acquisition:** The 360° video, encoded in ProRes 422 to maximize pixel fidelity, is decoded and split into individual frames using the OpenCV library in C++.
2. **Geometric Engine (C++):** Depending on the active branch, the engine handles the spatial topology differently:
   * **Main Branch (Cubemap):** Each frame undergoes *Backward Mapping*. A single target matrix ($3072 \times 1024$ pixels) is pre-allocated. The left, front, and right faces are populated directly in place using memory offsets via Regions of Interest (ROI). The `BORDER_WRAP` flag is applied to maintain toroidal continuity.
   * **Experimental Branch (HEALPix):** The engine natively samples the equirectangular frame into a 1D array of equal-area pixels using the HEALPix C++ library, preserving the exact spherical topology without seam distortions.
3. **Inter-Process Communication (IPC):** The Python environment leverages `subprocess.Popen` to instantiate the C++ engine as a child process. The raw binary stream (`stdout`) is ingested into a 1D `uint8` array via NumPy and explicitly duplicated (`.copy()`) to ensure mutability before deep learning ingestion.
4. **Inference Engine (PyTorch):** * **Main Branch (YOLO26):** Encapsulates the entire preprocessing, inference, and NMS pipeline within a unified `.predict()` method, outputting 2D bounding box tensors.
   * **Experimental Branch (Spherical U-Net):** The 1D HEALPix array is processed using the `healpy` Python library and a custom Spherical Transformer module to reconstruct local neighborhoods. A PyTorch U-Net architecture then performs pixel-perfect semantic segmentation directly on the continuous sphere.
5. **Post-Processing (Vectorization):** To avoid the severe performance penalty of the Python interpreter, VRAM tensors are brought to the CPU and cast to native integers in bulk using NumPy's C-backend vectorization (`.astype(int)`).

## 3. Hardware and Software Prerequisites

**Hardware Target**
* Developed and optimized on an MSI Vector 16 HX AI equipped with 32GB of system RAM.
* **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM) for Tensor Core acceleration.

**Software Dependencies**
* NVIDIA CUDA Toolkit and cuDNN.
* **C++ Environment:** `g++-13` (supporting C++20 and C++23 standards), CMake, HEALPix C++ API, and OpenCV C++ API.
* **Python Environment:** Python 3.x with a dedicated virtual environment (`venv`).
* **Core Python Libraries:** `torch`, `numpy`, `opencv-python`, `ultralytics` (for the Main Branch), and `healpy` (for the Experimental Branch).

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
pip install torch torchvision numpy opencv-python ultralytics healpy

```

## 5. Usage

The architecture handles the IPC internally within Python. The Python script autonomously launches the compiled C++ binary and manages the buffer.

```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# Main Branch: Execute the YOLO26 Engine 
python3 yolo26.py /path/to/your/video.mp4

# Experimental Branch: Execute the Spherical Semantic Segmentation Engine
python3 spherical_unet.py /path/to/your/video.mp4

```

## 6. License and Acknowledgments

This project is licensed under the **GNU AGPL-3.0 License**. See the `LICENSE` file for more details.

**Acknowledgments:**

* The core object detection baseline relies on the open-source architectures provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
* The experimental branch for resolving *Seam Discontinuity* and *Root-Point Anchoring* is heavily inspired by the **[Spherical Transformer](https://arxiv.org/pdf/2101.03848)** methodology proposed by Liu et al. (2021) and leverages the [healpy](https://github.com/healpy/healpy) package for topological sphere pixelization.