# Cheating-Detection-with-Jetson-Nano

# Exam Cheating Detection System on Jetson Nano
Visit the project on [Cheating]([https://github.com/yourusername/cheating-detection-nano](https://drive.google.com/file/d/1X7EtB3rxlj68vcwP_OnPwhwmUAUtWurq/view?usp=sharing))

## 📌 Introduction

This is a real-time exam cheating detection system implemented entirely on a **Jetson Nano** device. The system uses:

- A **camera** to capture video from the exam room
- **trt_pose** to perform human pose estimation
- **XGBoost** to classify cheating behaviors based on body keypoint coordinates

It can detect suspicious behaviors such as turning sideways, looking backward, or leaning excessively toward other candidates.

---

## 🧱 System Architecture

1. **Video Input:** A camera connected to Jetson Nano captures video during the exam.
2. **Pose Estimation:** `trt_pose` extracts 18 keypoints from detected human poses in real time.
3. **Feature Extraction:** Converts keypoint coordinates into a numerical feature vector.
4. **Classification:** A pre-trained **XGBoost** model classifies the behavior as cheating or not.
5. **Logging/Output:** Suspicious frames are saved or flagged, and detection results are displayed or logged.

---

## 🧰 System Requirements

- Jetson Nano 4GB with **JetPack 4.4**
- Python 3.6.9 or 3.7.5
- PyTorch 1.6.0
- TorchVision 0.6.0
- trt_pose (from NVIDIA)
- XGBoost
- OpenCV
- NumPy

---

## ⚙️ Installation

