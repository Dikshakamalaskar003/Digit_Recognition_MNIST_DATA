# DataSpeak – Neural Network Implementation from Scratch

## Project Overview

This project presents a **pure implementation of a feedforward neural network** using **Python and NumPy**, developed **solely for understanding the internal working of neural networks**.

The primary objective of this project is **conceptual clarity**, not performance optimization.  
No deep learning frameworks (such as TensorFlow or PyTorch) are used for model training or learning logic.

The project is designed as an **academic learning exercise** for **M.Sc. level study** to understand:
- Neural network architecture
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization

---

## Academic Purpose

- This project is **purely for understanding neural networks**
- Focus is on **mathematical and algorithmic implementation**
- Not intended to compete with optimized deep learning models
- Emphasizes **learning fundamentals over accuracy**

---

## Technologies Used

- Python 3  
- NumPy  
- TensorFlow (used **only** for loading the MNIST dataset)

---

## File Description

| File Name | Description |
|----------|-------------|
| `dataspeak.py` | Complete neural network implementation, training logic, and evaluation |

---

## Neural Network Architecture

The network follows a **fully connected feedforward structure**:


Example configurations:
- `[2, 2, 1]` – basic concept testing
- `[784, 50, 10]` – MNIST handwritten digit classification

---

## Core Components

### 1. Weight and Bias Initialization
- Weights initialized using random values scaled by input size
- Biases initialized for all layers except input

---

### 2. Activation Function
- Sigmoid activation function is used
- Introduces non-linearity into the network

---

### 3. Forward Propagation
- Computes weighted sums
- Applies activation function layer by layer
- Produces final output prediction

---

### 4. Training Method
- Stochastic Gradient Descent (SGD)
- Mini-batch based learning
- Multiple epochs of training

---

### 5. Backpropagation Algorithm
- Calculates output error
- Propagates error backward
- Computes gradients for weights and biases
- Updates parameters to minimize error

---

### 6. Evaluation
- Model predictions compared with actual labels
- Accuracy calculated as number of correct predictions

---

## MNIST Dataset Handling

- Input images of size `28 × 28` are flattened to `784 × 1`
- Pixel values normalized between 0 and 1
- Output labels converted into one-hot encoded vectors

---

## Results

- The model successfully learns basic patterns
- Achieves reasonable accuracy on MNIST
- Accuracy varies based on random initialization and training parameters

---

## Key Learning Outcomes

- Strong understanding of neural network fundamentals
- Practical implementation of backpropagation
- Clear insight into gradient descent optimization
- Ability to understand and debug neural network training logic
- Foundation for using advanced deep learning frameworks

---

## Limitations

- Uses sigmoid activation for all layers
- No advanced optimizers (Adam, RMSProp)
- No regularization or dropout
- Designed for learning, not production use

---

## Conclusion

This project serves as a **foundational neural network implementation** that demonstrates how learning occurs at a mathematical and algorithmic level.  
It is well-suited for **M.Sc. academic evaluation and conceptual understanding** of machine learning principles.

---

## Author

**Diksha Kamalaskar**  
M.Sc. Computer Science  

