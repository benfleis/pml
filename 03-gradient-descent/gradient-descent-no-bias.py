#!/usr/bin/env python3

# %% Cell 1
import numpy as np

# %% Cell 2
def predict(X, w, b):
    return X * w + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w, b):
    return 2 * np.average(X * (predict(X, w, b) - Y))


def train(X, Y, iterations, lr):
    w, b = 0, 0
    for i in range(iterations):
        print(f'Iteration {i:4d} => Loss: {loss(X, Y, w, b):.10f}')
        w -= gradient(X, Y, w) * lr
    return w

# %% Cell 3
X, Y = np.loadtxt("../02-pizza/pizza.txt", skiprows=1, unpack=True)

# %% Cell 4
def plot_it(w=None, b=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    x_max, y_max = 50, 50
    sns.set()
    plt.axis([0, x_max, 0, y_max])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Reservations", fontsize=12)
    plt.ylabel("Pizzas", fontsize=12)
    plt.plot(X, Y, "bo")
    if w is not None:
        plt.plot([0, x_max], [b or 0, x_max*w], 'blue', linewidth=1)
    plt.show()

# %% Cell 5
b = 0
w = train(X, Y, iterations=100, lr=0.001)
print(f'w={w:.3f}, b={b:.3f}')

x = 20
print(f'Prediction: x={x} => y={predict(x,w,b):.2f}')
plot_it(w, b)
