# -*- coding: utf-8 -*-#

import numpy as np
import random


class SimpleNetwork:

    def __init__(self, sizes):
        """
        sizes = [input_neurons, hidden_neurons, output_neurons]
        Example: [2, 3, 1]
        """
        self.sizes = sizes
        self.num_layers = len(sizes)

        # Random biases for all layers except input
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Random weights connecting each layer
        # example: weight matrix between 2 -> 3 will be shape (3,2)
        self.weights = [np.random.randn(y, x)/ np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # def sigmoid_prime(self, z):
    #     #Derivative of sigmoid
    #     s = self.sigmoid(z)
    #     return s * (1 - s)


    # Forward pass

    def feedforward(self, a):
        """
        Pass input 'a' through all layers.
        a is column vector (n,1)
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a


    # TRAINING (STOCHASTIC GRADIENT DESCENT)

    def train(self, training_data, epochs, batch_size, learning_rate):

        for epoch in range(epochs):

            random.shuffle(training_data)

            # Break dataset into batches
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, len(training_data), batch_size)
            ]

            for batch in batches:
                self.update_batch(batch, learning_rate)

            print(f"Epoch {epoch} finished")




    def update_batch(self, batch, learning_rate):

        # Initialize total gradients (start with zero)
        total_grad_b = [np.zeros(b.shape) for b in self.biases]
        total_grad_w = [np.zeros(w.shape) for w in self.weights]

        # Sum gradients from each training example
        for x, y in batch:
            grad_b, grad_w = self.backprop(x, y)

            total_grad_b = [nb + dnb for nb, dnb in zip(total_grad_b, grad_b)]
            total_grad_w = [nw + dnw for nw, dnw in zip(total_grad_w, grad_w)]

        # Apply gradient descent update rule
        self.weights = [
            w - (learning_rate / len(batch)) * nw
            for w, nw in zip(self.weights, total_grad_w)
        ]

        self.biases = [
            b - (learning_rate / len(batch)) * nb
            for b, nb in zip(self.biases, total_grad_b)
        ]

    def backprop(self, x, y):

        # Gradients (same shape as weights/biases)
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        #STEP 1: FORWARD PASS
        activation = x          # input layer
        activations = [x]       # store all activations
        zs = []                 # store all weighted sums (z)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        #STEP 2: OUTPUT ERROR
        # delta = (output - expected) * sigmoid'
        #delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        delta = (activations[-1] - y)

        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)

        # STEP 3: BACKWARD PASS
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp=self.sigmoid_prime(z)
            # propagate error backward
            delta = np.dot(self.weights[-layer + 1].T, delta) * sp

            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta, activations[-layer - 1].T)

        return grad_b, grad_w


    def evaluate(self, test_data):
        """
        Returns how many test samples were predicted correctly.
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(pred == actual) for (pred, actual) in results)

net = SimpleNetwork([2, 2, 1])

x = np.array([[1],
              [0]])

output = net.feedforward(x)
print("Output =", output)

training_data = [
    (np.array([[1],[0]]), np.array([[1]])),
    (np.array([[0],[1]]), np.array([[0]])),
]

net.train(training_data, epochs=10, batch_size=1, learning_rate=0.5)

result = net.evaluate(training_data)
print("Correct predictions:", result)

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def reshape_image(img):
    # flatten 28x28 → 784x1 and normalize
    return img.reshape(784, 1) / 255.0

def one_hot(label):
    y = np.zeros((10, 1))
    y[label] = 1.0
    return y

training_data = [
    (reshape_image(img), one_hot(label))
    for img, label in zip(train_images, train_labels)
]

test_data = [
    (reshape_image(img), one_hot(label))
    for img, label in zip(test_images, test_labels)
]

net = SimpleNetwork([784, 50, 10])

net.train(
    training_data=training_data,
    epochs=30,
    batch_size=32,
    learning_rate=0.5
)

correct = net.evaluate(test_data)
accuracy = (correct / len(test_data)) * 100
print(f"Accuracy: {accuracy:.2f}%")

