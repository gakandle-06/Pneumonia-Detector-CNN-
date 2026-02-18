# 🫁 Pneumonia Detection from Chest X-Rays using CNN

A deep learning project that detects and classifies pneumonia from chest X-ray images. This model distinguishes between **Normal**, **Viral Pneumonia**, and **Bacterial Pneumonia** cases using a Convolutional Neural Network (CNN).

## 📌 Project Overview
Pneumonia is a life-threatening infectious disease affecting the lungs. Early and accurate diagnosis is crucial for effective treatment. This project uses computer vision to analyze chest X-rays and assist medical professionals by automating detection.

Key features of this implementation:
* **Custom Data Balancing**: Addresses class imbalance (Bacterial > Viral/Normal) using targeted image augmentation.
* **Medical-Grade Augmentation**: Uses specific geometric transformations (without horizontal flipping) to preserve anatomical correctness (e.g., heart position).
* **Performance Visualization**: Includes confusion matrices and prediction confidence heatmaps.

## 📂 Dataset
The dataset is sourced from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset)

* **Total Images**: ~5,800+ X-Rays
* **Classes**:
    * `NORMAL`: Healthy lung tissue.
    * `PNEUMONIA (Bacteria)`: Characterized by focal lobar consolidation.
    * `PNEUMONIA (Virus)`: Characterized by diffuse interstitial patterns.

> **Note**: The original dataset is imbalanced. This project includes a preprocessing pipeline that generates synthetic variations for minority classes to ensure unbiased training.

## 🛠️ Tech Stack
* **Core**: Python 3.x
* **Deep Learning**: TensorFlow, Keras
* **Data Processing**: NumPy, Pandas, OpenCV
* **Visualization**: Matplotlib, Seaborn

## ⚙️ Key Implementation Details

### 1. Data Balancing & Augmentation
The script automatically detects the class with the highest count (Bacterial) and generates augmented images for the minority classes (Normal, Viral) to achieve an approximate 1:1:1 ratio.
* *Techniques used*: Rotation, Width/Height Shift, Shear, Zoom, and Brightness adjustments.
* *Constraint*: `horizontal_flip=False` is strictly enforced to maintain correct organ placement (e.g., the heart must remain on the left).

### 2. Model Architecture
The model is a Sequential CNN designed for feature extraction and classification:
* **Convolutional Layers**: Extract features like edges, textures, and opacities.
* **Max Pooling**: Reduces dimensionality and computational cost.
* **Dropout**: Prevents overfitting.
* **Dense Layers**: Final classification (Softmax activation).
