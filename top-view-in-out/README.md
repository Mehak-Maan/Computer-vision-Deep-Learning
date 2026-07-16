# 🚶‍♂️ Top-View In/Out Person Counter

A highly efficient computer vision application designed to monitor foot traffic from a top-down camera perspective. This project is specifically architected for deployment on the **Raspberry Pi**, ensuring edge-computing efficiency and real-time analytical capabilities.

## 🌟 Key Features

* **Native Raspberry Pi Support:** Utilizes `picamera2` directly, mapping the BGR888 feed to OpenCV without expensive format conversions, maximizing FPS on the Pi.
* **Edge Model Acceleration:** Uses advanced model exports like **OpenVINO** and **NCNN** (`inout_ncnn_model.7z`) to drastically reduce inference times on low-end CPUs and ARM architectures.
* **Robust Custom Tracker:** Implements a custom, lightweight centroid-based tracker with hysteresis patience (keeps IDs alive across missed frames) to handle top-down visual complexities where subjects warp or blend.
* **Smart Directional Counting:** 
  - Defines precise horizontal crossing zones (`UP_LIMIT` and `DOWN_LIMIT`).
  - Utilizes Debounce Delays and strict state-transition logic (`TOP -> BOTTOM`, `BOTTOM -> TOP`) to prevent double-counting when subjects linger near the threshold.
* **Live Telemetry & UI:** Draws clean tracking boxes (via `cvzone`), trajectory points, entry/exit counters, and live FPS metrics directly on the video feed.

## 🛠️ Hardware & Software Stack
* **Hardware:** Raspberry Pi (Mounted overhead).
* **Models:** YOLOv8 trained specifically for top-down human geometry, exported to OpenVINO and NCNN.
* **Libraries:** Ultralytics, OpenCV, cvzone, Picamera2.

## 🚀 How to Run

1. Mount your Raspberry Pi Camera in a top-down orientation facing the floor/doorway.
2. Adjust the `LIMIT_LINE_Y` variable in the code if your camera angle differs.
3. Start the application:
```bash
python Best.py
```
*(Press 'ESC' to terminate the visual feed and process).*
