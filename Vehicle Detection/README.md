# 🚗 Smart Vehicle & ANPR System (Live Parking Edge Solution)

This project provides an end-to-end, edge-computed **Automatic Number Plate Recognition (ANPR)** and vehicle logging system. Built to run seamlessly on a **Raspberry Pi**, it captures live video feeds, tracks vehicles, extracts license plate text, and updates an interactive local dashboard and an IoT Cloud.

## 🌟 Key Features

* **Raspberry Pi Hardware Acceleration:** Fully integrated with `picamera2` to capture high-framerate RGB streams directly from the Pi Camera Module.
* **Dual-Model Inference pipeline:** 
  1. Detects and tracks vehicles (Cars, Trucks, Buses, Motorcycles) using `vehicle.onnx`.
  2. Crops the vehicle and runs a secondary detection using `numplate.onnx` to isolate the license plate.
* **Optical Character Recognition (OCR):** Uses `easyocr` within a background threaded queue (to prevent frame blocking) to read alphanumeric characters from cropped plates.
* **Smart Parking Logic:** Features stateful memory (`parking_memory.json`) that manages Entry/Exit events based on debounce timers, preventing duplicate logs and tracking active parking durations.
* **Live HTML Report Generation:** Dynamically generates a beautiful, stylized local web report (`parking_report_live.html`) containing vehicle snapshots, plate badges, entry/exit timestamps, and current parking status.
* **Cloud Telemetry:** Pushes real-time JSON payloads (Vehicle Type, Plate, Status, Timestamps) to an OpenRemote IoT dashboard via REST APIs.

## 🛠️ Hardware & Software Stack
* **Target Hardware:** Raspberry Pi (3/4/5) with Pi Camera Module.
* **Deep Learning:** Ultralytics YOLO (ONNX format for CPU edge acceleration).
* **Libraries:** OpenCV, EasyOCR, threading, requests, Picamera2.

## 🚀 Quick Start
Run the live tracking pipeline on your Raspberry Pi:
```bash
python live_detection.py
```
The system will automatically initialize the camera, load the ONNX models, and begin processing the live feed while updating `parking_report_live.html` in real-time.
