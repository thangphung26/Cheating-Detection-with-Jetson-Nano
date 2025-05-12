# Exam Cheating Detection System on Jetson Nano

## 📌 Introduction

This project is a real-time exam cheating detection system running entirely on a **Jetson Nano**. It uses:

- A **camera** to capture student behavior
- [`trt_pose`](https://github.com/NVIDIA-AI-IOT/trt_pose) for real-time pose estimation
- [`XGBoost`](https://xgboost.readthedocs.io/) for behavior classification (cheating vs non-cheating)

👉 Visit the project on [Google Drive](https://drive.google.com/file/d/1X7EtB3rxlj68vcwP_OnPwhwmUAUtWurq/view?usp=sharing)

---

## 🧱 System Architecture

1. **Camera Feed:** Captures video of the exam environment
2. **Pose Estimation:** `trt_pose` detects 18 body keypoints for each person
3. **Feature Extraction:** Coordinates are flattened into a feature vector
4. **Classification:** XGBoost detects whether behavior indicates cheating
5. **Result:** Suspicious actions are flagged or saved as images

---

## 💻 Requirements

- **Jetson Nano (4GB)**
- **JetPack 4.4 (L4T R32.4.3)**
- Python 3.6.9 or 3.7.5
- PyTorch 1.6.0
- TorchVision 0.6.0
- OpenCV
- NumPy
- XGBoost
- trt_pose

---

## ⚙️ Installation

### 1. Install JetPack 4.4 on Jetson Nano
- Download JetPack 4.4 image from [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/jetpack-archive)
- Flash the image to SD card using [balenaEtcher](https://www.balena.io/etcher/)
- Insert SD card into Jetson Nano and boot up

### 2. Setup Environment
- install [`trt_pose`](https://github.com/NVIDIA-AI-IOT/trt_pose) for real-time pose estimation
- install numpy,xgboost,pytorch,cuda,torchvision,...
### 3. Clone Project

