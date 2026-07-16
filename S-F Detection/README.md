# 🔥 Smoke and Fire Detection for Edge Devices

A mission-critical **Smoke and Fire (S-F) Detection** computer vision pipeline, optimized for real-time monitoring and early warning alerts. This system is heavily optimized to run on low-power Edge computing devices, primarily the **Raspberry Pi**.

## 🌟 Key Features

* **Cross-Platform & Hardware Ready:** Includes a flexible inference script (`yolo_detect.py`) capable of ingesting video from generic files, USB webcams, and natively from the Raspberry Pi Camera via `picamera2`.
* **Aggressive Edge Optimizations:** Contains models exported to multiple lightweight formats:
  - **ONNX** (`fire.onnx`, `slim_fire.onnx`) for CPU/GPU generic execution.
  - **NCNN** (`fire_ncnn_model.7z`) specifically tailored for high-speed inference on ARM-based architectures like the Raspberry Pi.
* **Flexible Input Arguments:** Easily run inference via CLI arguments controlling confidence thresholds, input sources, display resolution, and video recording.
* **Real-time FPS Calculation:** In-built telemetry to monitor the pipeline's Frames Per Second (FPS) to ensure real-time capabilities on constrained hardware.

## 🛠️ Hardware & Software Stack
* **Hardware:** Raspberry Pi (Camera Module via `picamera2`) or standard x86 PC.
* **Model Formats:** YOLO PyTorch (`.pt`), ONNX, and NCNN.
* **Tech Stack:** OpenCV, Ultralytics, Argparse.

## 🚀 How to Run
You can run the detection script on various sources:

**Run on a Raspberry Pi using the Pi Camera:**
```bash
python yolo_detect.py --model fire.onnx --source picamera0 --thresh 0.45
```

**Run on a standard USB Webcam:**
```bash
python yolo_detect.py --model fire.pt --source usb0 --thresh 0.5
```

**Run on a saved video and record the output:**
```bash
python yolo_detect.py --model fire.onnx --source testvid.mp4 --resolution 640x480 --record
```
