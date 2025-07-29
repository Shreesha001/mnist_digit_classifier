
# MNIST Digit Classifier

A simple and effective handwritten digit classification model trained on the MNIST dataset using TensorFlow and Keras. This project demonstrates core deep learning concepts such as dense layers, activation functions, softmax, and categorical cross-entropy loss.

---

## Overview

This notebook trains a neural network to classify grayscale images of handwritten digits (0–9). Each image is 28×28 pixels and belongs to one of 10 classes. The final model achieves high accuracy on unseen data and can be used to predict digits from new inputs.

---

## Model Architecture

- **Input Layer**: Flattened 28×28 image (784 features)
- **Hidden Layers**:
  - Dense layer with 128 neurons and ReLU activation
  - Dense layer with 64 neurons and ReLU activation
- **Output Layer**:
  - Dense layer with 10 units (one per digit) and softmax activation

---

## Key Components

- **Loss Function**: `SparseCategoricalCrossentropy`
- **Optimizer**: `Adam`
- **Metrics**: Accuracy
- **Activation Functions**: `ReLU` for hidden layers, `Softmax` for output
- **Evaluation**: Includes model accuracy on the test set and manual prediction with softmax probabilities

---

## Results

- **Training Accuracy**: ~97%
- **Test Accuracy**: ~97%
- The model generalizes well on unseen handwritten digits.

---

## Sample Prediction Logic

```python
# Pick one test sample
index = 0
sample = x_test[index].reshape(1, -1)

# Get logits (pre-softmax scores)
logits = model(sample)

# Apply softmax manually to convert to probabilities
probs = tf.nn.softmax(logits).numpy()

# Predicted digit
predicted_class = np.argmax(probs)
```

---

## Dataset

- **Source**: [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- **Shape**: 60,000 training samples, 10,000 test samples
- **Preprocessing**: Pixel values normalized to the range [0, 1]

---

