# Neural Network from Scratch for MNIST Digit Classification

This project implements a **fully connected neural network from scratch**, using only **NumPy**, to classify handwritten digits from the MNIST dataset. It demonstrates core machine learning principles like **forward propagation**, **backpropagation**, and **gradient descent**, without relying on any ML libraries like TensorFlow or PyTorch.

---

## Features

- Built a 2-layer neural network entirely from scratch using NumPy
- Achieved **85% accuracy** on the MNIST dataset
- Implemented:
  - ReLU and softmax activations
  - One-hot encoding for label representation
  - Manual forward and backward propagation
  - Weight and bias updates using gradient descent

---

## Learning Rate Strategy

To accelerate early learning:
- Initially set the **learning rate `alpha = 0.15`** (higher than the standard 0.10)
- Applied a **learning rate decay strategy** where:
  - The learning rate **decreased gradually per iteration**
  - Eventually stabilized at **alpha = 0.10**
  
This helped the model learn quickly in the beginning without overshooting the optimal parameters during later training epochs.

---

## File Overview

| File                     | Description                             |
|--------------------------|-----------------------------------------|
| `mnist-nn-python.ipynb`  | Jupyter Notebook with full implementation and training logs |
| `README.md`              | Project overview and setup instructions |

---

## Dataset

- Used the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer)
- Preprocessed:
  - Flattened 28×28 images into 784-length vectors
  - Normalized pixel values to range [0, 1]

---

## Training & Accuracy

- Reached 85%+ accuracy within 500 iterations on training data
- Monitored accuracy every 10 iterations
- Model performance was stable after ~400 iterations

---

## Training Progress

Using a decaying learning rate strategy (α from 0.15 → 0.10), the model achieved:

- Iteration 0:  Accuracy = 13.2%
- Iteration 100: Accuracy = 61.5%
- Iteration 300: Accuracy = 81.5%
- Iteration 490: Accuracy = 85.2%

This learning rate schedule helped the model converge faster in the early stages while maintaining stability in later epochs.

---

## How to Run

Make sure you have Python and NumPy installed:

```bash
pip install numpy pandas

## 📚 Inspiration

This project was based on the excellent tutorial by Samson Zhang (https://www.youtube.com/watch?v=w8yWXqWQYmU&t).

I improved the project by implementing the faster initial learning rate strategy
