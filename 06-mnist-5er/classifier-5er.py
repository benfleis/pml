#!/usr/bin/env python3

# %%
import numpy as np
import mnist as data

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):
    return np.round(forward(X, w))

def loss(X, Y, w):
    y_hat = forward(X, w)
    term_0 = Y * np.log(y_hat)
    term_1 = (1 - Y) * np.log(1 - y_hat)
    return -np.average(term_0 + term_1)

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print(f'Iteration {i:4d} => Loss: {loss(X, Y, w):.20f}')
        w -= gradient(X, Y, w) * lr
    return w

def test(X ,Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_rate = float(correct_results) / total_examples
    #print(f'Weights: {w.T[0]}')
    print(f'Success: {correct_results}/{total_examples} ({success_rate:.3f})')


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print(f'Iteration {i:4d} => Loss: {loss(X, Y, w):.20f}')
        w -= gradient(X, Y, w) * lr
    return w

w = train(data.X_train, data.Y_train, iterations=100, lr=1e-5)
test(data.X_test, data.Y_test, w)
