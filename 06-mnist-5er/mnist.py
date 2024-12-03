#!/usr/bin/env python3

# %%
import gzip
import numpy as np
import struct

# %%

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        # read header
        _, image_cnt, cols, rows = struct.unpack('>IIII', f.read(16))
        # then all pixels
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # reshape into a matrix where each line is an image
        return pixels.reshape(image_cnt, cols * rows)

def prepend_bias(X):
    # insert a col of 1s in the position 0 of X
    # ('axis=1' -> "insert a col, not a row")
    return np.insert(X, 0, 1, axis=1)

# 60000 images, each 785 elts (1 bias + 28*28 pixels)
X_train = prepend_bias(load_images("./data/mnist/train-images-idx3-ubyte.gz"))

# 10000 images, each 785 elts, same structure as X_train
X_test = prepend_bias(load_images("./data/mnist/t10k-images-idx3-ubyte.gz"))

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = f.read(8) # skip hdr
        labels = f.read() # all the labels are belong to us.
        return np.frombuffer(labels, dtype=np.uint8).reshape(-1, 1)

def encode_fives(Y):
    return (Y == 5).astype(int)

# 60k labels; 5 -> 1, [0-4,6-9] -> 0
Y_train = encode_fives(load_labels("./data/mnist/train-labels-idx1-ubyte.gz"))

# 60k labels; 5 -> 1, [0-4,6-9] -> 0
Y_test = encode_fives(load_labels("./data/mnist/t10k-labels-idx1-ubyte.gz"))
