import numpy as np
import pandas as pd

def one_hot_encode(y_labels):
    classes = np.unique(y_labels)
    y_int = np.array([np.where(classes == label)[0][0] for label in y_labels])
    Y = np.zeros((len(y_int), len(classes)))
    Y[np.arange(len(y_int)), y_int] = 1
    return Y, classes


def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def split_data(X, Y, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    
    return (X[train_idx], Y[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx])


def confusion_matrix_np(Y_true, Y_pred, classes):
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(Y_pred, axis=0)
    K = len(classes)
    M = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return pd.DataFrame(M, index=classes, columns=classes)
