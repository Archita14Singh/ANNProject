# ANN Binary Classification – Breast Cancer Prediction

## Problem Statement
The goal of this project is to build an Artificial Neural Network (ANN)
to classify whether a breast tumor is malignant or benign using real
medical diagnostic data.

## Dataset
- Source: Breast Cancer Wisconsin Dataset (scikit-learn)
- Type: Real-world binary classification dataset
- Features: Cell nucleus measurements such as radius, texture, area, etc.
- Target:
  - 0 → Benign
  - 1 → Malignant

## Approach
- Loaded dataset using `load_breast_cancer`
- Performed train-test split
- Applied feature scaling using StandardScaler
- Built a multi-layer ANN for binary classification
- Used Sigmoid activation in the output layer
- Evaluated the model using accuracy and confusion matrix

## Model Architecture
- Input Layer: 30 features
- Hidden Layer 1: Dense (32 neurons, ReLU)
- Hidden Layer 2: Dense (16 neurons, ReLU)
- Output Layer: Dense (1 neuron, Sigmoid)

## Training Details
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 50
- Batch Size: 32

## Visualizations
- Training vs Validation Loss
- Training vs Validation Accuracy
- Confusion Matrix for prediction evaluation

## Key Concepts Covered
- Neuron and Perceptron implementation
- Activation functions (ReLU, Sigmoid)
- Binary classification using ANN
- Overfitting vs underfitting analysis
- Probability-based predictions

## Results
- Model achieved strong accuracy on test data
- Clear distinction between malignant and benign cases
- Performance evaluated using precision, recall, and F1-score

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

## Conclusion
This project provides a complete end-to-end implementation of an ANN
for binary classification and strengthens understanding of core deep
learning concepts.
