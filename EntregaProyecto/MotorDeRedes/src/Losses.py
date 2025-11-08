import numpy as np

def categorical_cross_entropy(Y_true, Y_pred):
    m = Y_true.shape[0]
    eps = 1e-8
    loss = -np.mean(np.sum(Y_true * np.log(Y_pred.T + eps), axis=1))
    return loss


def mean_squared_error(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred.T) ** 2)
