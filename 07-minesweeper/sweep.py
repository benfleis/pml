#!/usr/bin/env python3

# %%
import typing
import numpy as np
import data

# sanity check
assert data.train.shape == (160, 61) # 160 samples x (60 dims + 1 label)
assert data.test.shape == (48, 61) # 48 samples x (60 dims + 1 label)

# %%
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

def encode_one_hot(Y_labels, label_values):
    Y = np.zeros((Y_labels.shape[0], len(label_values)))
    for i, label in enumerate(Y_labels):
        Y[i][label_values[label]] = 1.0
    return Y

X_train = prepend_bias(data.train[:,:-1])
X_test = prepend_bias(data.test[:,:-1])

def encode_y_labels(values: np.ndarray, value_to_label: typing.Callable):
    labels = np.empty((values.shape[0]), dtype=np.str_)
    values = np.rint(values, np.empty(values.shape[0], dtype=np.int32), casting='unsafe')
    for (i, value) in enumerate(values):
        labels[i] = value_to_label(value)
    return labels

Y_train_labels = encode_y_labels(data.train[:,-1], data.labels_ordered.__getitem__)
Y_train_hot = encode_one_hot(Y_train_labels, data.label_values)
Y_test_labels = encode_y_labels(data.test[:,-1], data.labels_ordered.__getitem__)
Y_test_hot = encode_one_hot(Y_test_labels, data.label_values)

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
    return -np.sum(term_0 + term_1) / X.shape[0]

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def report(X_train, Y_train, X_test, Y_test, i, w):
    train_oks, _, train_count = test(X_train, w, Y_train)
    #import IPython.terminal.debugger; IPython.terminal.debugger.set_trace()
    test_oks, _, test_count = test(X_test, w, Y_test)

    train_rate = f"{train_oks}/{train_count}={(train_oks/train_count):.4f}"
    test_rate = f"{test_oks}/{test_count}={(test_oks/test_count):.4f}"
    print(f"iter {i:6d}: TRAIN loss={loss(X_train, Y_train, w):6f} rate={train_rate}; TEST rate={test_rate}")


def train(iters: int, lr: float, X_train, Y_train, X_test, Y_test):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iters):
        report(X_train, Y_train, X_test, Y_test, i, w)
        g = gradient(X_train, Y_train, w)
        w -= lr * g
    report(X_train, Y_train, X_test, Y_test, iters, w)
    return w

def test(X, w, Y):
    y_hat = np.round(forward(X, w))
    oks = int(np.count_nonzero(Y == y_hat) / Y.shape[1])
    errs = Y.shape[0] - oks
    return (oks, errs, Y.shape[0])

# %%

train(100000, 0.01, X_train, Y_train_hot, X_test, Y_test_hot)
