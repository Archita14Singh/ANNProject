# ANN Regression – House Price Prediction

## Problem Statement
The objective of this project is to build an Artificial Neural Network (ANN)
to predict house prices based on multiple input features. This project helps
in understanding how ANN can be applied to regression problems.

## Dataset
- Source: California Housing Dataset (scikit-learn)
- Type: Real-world regression dataset
- Features include:
  - Median income
  - House age
  - Average rooms
  - Population
  - Latitude & longitude
- Target variable: Median house value

## Approach
- Loaded dataset using `fetch_california_housing`
- Performed train-test split
- Applied feature scaling using StandardScaler
- Built a multi-layer ANN using TensorFlow/Keras
- Trained the model using Mean Squared Error loss
- Evaluated performance using loss metrics

## Model Architecture
- Input Layer: 8 features
- Hidden Layer 1: Dense (ReLU activation)
- Hidden Layer 2: Dense (ReLU activation)
- Output Layer: Dense (Linear activation)

## Key Concepts Covered
- ANN for regression problems
- Importance of feature scaling
- Activation functions (ReLU)
- Loss function: Mean Squared Error
- Forward propagation and backpropagation

## Results
- Model successfully learned non-linear patterns in housing data
- Training and validation loss curves were analyzed
- Predictions were generated on unseen test data

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Matplotlib

## Conclusion
This project demonstrates how Artificial Neural Networks can be effectively
used for regression tasks and serves as a foundation for more advanced deep
learning models.
