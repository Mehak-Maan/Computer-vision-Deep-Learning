# 👷‍♂️ Construction PPE (Personal Protective Equipment) Detection

A safety-critical computer vision system utilizing state-of-the-art YOLO architectures (YOLOv8 and YOLO11) to monitor construction sites. It ensures workers are wearing necessary protective gear such as hard hats and high-visibility vests.

## 🌟 Key Features
* **Multi-Generational YOLO Comparison:** Includes comprehensive training and inference pipelines for both YOLOv8 and the bleeding-edge YOLO11 architectures, allowing for performance and accuracy benchmarking.
* **Scale Variations:** Features notebooks testing different model sizes (e.g., `yolov8s` vs `yolov8` standard, `yolov11s` vs `yolov11`) to balance FPS and mean Average Precision (mAP) for edge deployment.
* **Video Analytics:** Includes a test video (`indianworkers.mp4`) to validate the model's tracking and detection consistency across moving frames in real-world construction environments.
* **Custom Datasets:** Structured to ingest custom bounding-box datasets specifically tailored for construction equipment.

## 🛠️ Hardware & Software Stack
* **Tech Stack:** Ultralytics (YOLOv8/YOLO11), PyTorch, OpenCV.
* **Deployment:** Can be deployed on GPU servers or optimized (via ONNX/TensorRT) for local site edge devices.

## 🚀 Usage
Execute the Jupyter notebooks (`Construction_detection_yolov8s.ipynb`, `construction_detection_yolo11s.ipynb`, etc.) to review the training metrics, view augmented batches, and run inference on `indianworkers.mp4`.
