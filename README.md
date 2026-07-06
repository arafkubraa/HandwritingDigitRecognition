# Handwritten Digit Recognition

<p align="center">
  <img src="images/hand1.png" width="100%" alt="Handwritten Digit Recognition">
</p>

> A desktop application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue)

</p>

---

# Overview

Handwritten Digit Recognition is a desktop application that classifies handwritten digits using a Convolutional Neural Network (CNN).

The application enables users to draw digits on an interactive canvas or load an image for prediction. Images are preprocessed and fed into a trained deep learning model, which predicts the corresponding digit in real time.

The project demonstrates the complete machine learning workflow, including preprocessing, model training, inference, and graphical user interface integration.

---

# Features

- Handwritten digit recognition
- CNN-based deep learning model
- MNIST dataset support
- Real-time prediction
- Image preprocessing
- Interactive drawing canvas
- Image upload
- Prediction confidence
- Desktop graphical interface
- Fast inference

---

# Technology Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- PIL (Pillow)

---


# Machine Learning Pipeline

```text
Input Image
      │
      ▼
Preprocessing
      │
      ▼
Normalization
      │
      ▼
CNN Model
      │
      ▼
Softmax Classification
      │
      ▼
Predicted Digit
```

---

# Model

The recognition model is based on a Convolutional Neural Network trained using the MNIST handwritten digit dataset.

Pipeline:

- Image preprocessing
- Normalization
- Feature extraction using convolutional layers
- Max pooling
- Fully connected layers
- Softmax classification

---

# Dataset

- MNIST
- 60,000 Training Images
- 10,000 Test Images
- Image Size: 28×28
- Classes: 10 (Digits 0–9)

---

# Performance

- High recognition accuracy on MNIST
- Fast prediction time
- Lightweight model suitable for desktop applications

---

# Future Improvements

- EMNIST alphabet recognition
- Custom dataset training
- Better preprocessing
- Model comparison
- GPU acceleration
- Export trained models
- Web version

---

# Developers

Developed by

- Kübra Atlan

---

# License

This repository is shared for portfolio and educational purposes.

The source code is publicly available for learning and research.
