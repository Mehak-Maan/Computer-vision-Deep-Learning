# 🌾 Sugarcane Variety Classification & Analysis

An agricultural technology (AgriTech) project utilizing Deep Learning (PyTorch) to analyze, detect, and classify different varieties of sugarcane. This model can be integrated into sorting pipelines or used by farmers via mobile camera feeds.

## 🌟 Key Features
* **Custom PyTorch Architecture:** Built directly on PyTorch, providing a high degree of flexibility for modifying the convolutional layers and fine-tuning the model (`sugarcane_model.pth`, ~98MB) on distinct sugarcane features.
* **Live Webcam Inference:** Includes a `webcam.ipynb` notebook to demonstrate real-time classification by piping the local camera feed directly into the PyTorch inference engine.
* **Data-Driven Configuration:** Uses `sugarcane_varieties.yaml.txt` to dynamically map output tensors to agricultural variety names, making it extremely easy to expand the model's repertoire without changing the core code.
* **Structured Analytics:** `sugarcane.ipynb` covers the complete machine learning lifecycle, from exploratory data analysis (EDA) of the `sugarcane_data` dataset to training loops, loss plotting, and validation metrics.

## 🛠️ Hardware & Software Stack
* **Tech Stack:** PyTorch, Torchvision, OpenCV, NumPy, PyYAML.
* **Deployment:** Can run locally on CPU for basic webcam inference or on CUDA-enabled GPUs for high-speed batch processing.

## 🚀 Usage
1. Open `sugarcane.ipynb` to explore the training methodology or to retrain the model on new sugarcane datasets.
2. Open `webcam.ipynb` to load the pre-trained `sugarcane_model.pth` and run live video classification using your local webcam.
