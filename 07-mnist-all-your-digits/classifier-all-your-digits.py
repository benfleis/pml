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
    y_hat = forward(X, w)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

def loss(X, Y, w):
    y_hat = forward(X, w)
    term_0 = Y * np.log(y_hat)
    term_1 = (1 - Y) * np.log(1 - y_hat)
    # was: -np.average(term_0 + term_1)
    return -np.sum(term_0 + term_1) / X.shape[0]

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def one_hot_encode(Y):
    labels_cnt = Y.shape[0]
    classes_cnt = 10 # reminder == number of digits
    encoded_Y = np.zeros((labels_cnt, classes_cnt))
    for i in range(labels_cnt):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    test_examples_cnt = Y_test.shape[0]
    matches = matches / test_examples_cnt
    training_loss = loss(X_train, Y_train, w)
    print(f'{iteration: 4d} - Loss: {training_loss:10f} Matches:{matches:3f}')

def train(iterations, lr, X_train, Y_train, X_test, Y_test):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iterations):
        #print(f'Iteration {i:4d} => Loss: {loss(X_train, Y_train, w):.20f}')
        report(i, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
    report(iterations, X_train, Y_train, X_test, Y_test, w)
    return w

def test(X ,Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_rate = float(correct_results) / total_examples
    #print(f'Weights: {w.T[0]}')
    print(f'Success: {correct_results}/{total_examples} ({success_rate:.3f})')

# %%
# 60k labels, single digits 0-9
Y_train_unencoded = data.load_labels("data/mnist/train-labels-idx1-ubyte.gz")

# 60k labels, -> one hot encoding
# 10k labels, digits 0-9
Y_test = data.load_labels("data/mnist/t10k-labels-idx1-ubyte.gz")

w = train(200, 1e-6, data.X_train, Y_train, data.X_test, Y_test)
