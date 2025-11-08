import numpy as np

class Activations:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dsigmoid(a):
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def drelu(a):
        return np.where(a > 0, 1, 0)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def dtanh(a):
        return 1 - np.square(a)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
