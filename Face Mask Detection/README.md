# 😷 Face Mask Detection (TFLite Optimized)

A computer vision classification pipeline designed to detect whether individuals are adhering to face mask protocols. It combines traditional computer vision techniques with modern convolutional neural networks, heavily optimized for edge devices and mobile platforms.

## 🌟 Key Features
* **Hybrid Detection Pipeline:** Utilizes a highly efficient `haarcascade_frontalface_default.xml` to quickly detect facial bounding boxes, followed by a CNN to classify the cropped face into "Mask" or "No Mask".
* **Edge-Optimized Models:** Contains both a standard Keras model (`face_mask_model.h5`) and a highly compressed TensorFlow Lite model (`face_mask_model.tflite`) for deployment on mobile devices or Raspberry Pi.
* **Complete Training Pipeline:** Provides structured datasets (`Train`, `Validation`, `Test`) and full training notebooks to retrain or fine-tune the model parameters.
* **Label Encoding:** Includes `label_encoder.pkl` to strictly map model output tensors to human-readable string labels.

## 🛠️ Hardware & Software Stack
* **Deployment Hardware:** Any CPU, Mobile Device (Android/iOS via TFLite), or Raspberry Pi.
* **Tech Stack:** TensorFlow/Keras, OpenCV, Scikit-learn (Label Encoding).

## 🚀 Usage
Run `Face Mask Detection.ipynb` to view the training process and basic inference.
Run `TFLite.ipynb` to see how the model is quantized and converted for Edge deployment.
