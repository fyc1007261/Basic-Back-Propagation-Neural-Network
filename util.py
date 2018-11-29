import numpy as np


def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))


def relu(arr):
    return (arr > 0) * arr


def tanh(arr):
    return np.tanh(arr)


if __name__ == "__main__":
    array = np.array([[1], [3], [-444], [2132]])
    print(sigmoid(array))