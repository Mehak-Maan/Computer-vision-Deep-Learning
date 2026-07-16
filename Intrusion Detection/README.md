# 🛡️ Intrusion Detection System (Edge AI & IoT Integrated)

A state-of-the-art **Intrusion Detection System** fully optimized for deployment on edge devices like the **Raspberry Pi**. It leverages advanced computer vision, specifically YOLO models, for real-time person detection, color analysis, and unauthorized zone breaching.

## 🌟 Key Features

* **Raspberry Pi Integration:** Natively uses `picamera2` library for optimized hardware-accelerated video capture on Raspberry Pi.
* **Smart Zone Management:** Users can dynamically draw polygonal intrusion zones via a local graphical interface or Web UI. The coordinates are saved persistently in `zone_config.json`.
* **Deep Learning Object Tracking:** Uses YOLO object detection combined with ByteTrack to track unique individuals across frames, even handling complex postures (crawling, sitting).
* **Automated Color Analysis:** Extracts bounding boxes of detected individuals, segments shirt and pant regions, and performs color inference to generate a descriptive profile of the intruder.
* **IoT & Cloud Sync (OpenRemote):** Integrates seamlessly with the OpenRemote IoT platform. It utilizes both REST API and MQTT to sync live data (intruder details, entry/exit timestamps, clothing color, and snapshot base64 URLs) to a centralized command center.
* **Local Web Dashboard (Flask):** Hosts a lightweight Flask server out of the box (`http://localhost:5000`) for remote monitoring, displaying a real-time MJPEG video feed, and maintaining a live tabular log of all intrusion events.
* **Robust Logging System:** Saves intruder timestamps, bounding box locations, and ID details to local text and JSON files (`intruder_movement.log`, `intrusion_stats.json`).

## 🛠️ Hardware & Software Stack
* **Hardware:** Raspberry Pi (Camera Module V2/V3 compatible)
* **Frameworks:** ultralytics (YOLO), OpenCV, Flask, Eclipse Paho MQTT, Shapely (for polygon logic)
* **Models:** Highly optimized PyTorch/NCNN models for smooth FPS on ARM architecture.

## 🚀 How to Run

1. Connect your Raspberry Pi camera and ensure I2C/Camera interfaces are enabled (`raspi-config`).
2. Install dependencies: `pip install opencv-python ultralytics flask paho-mqtt shapely picamera2`
3. Run the core system:
   ```bash
   python intrusion.py
   ```
4. Access the live dashboard by navigating to the Pi's IP address on port 5000 (e.g., `http://192.168.1.x:5000`).
