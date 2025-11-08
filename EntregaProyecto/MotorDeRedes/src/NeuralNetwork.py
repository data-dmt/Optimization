import numpy as np
from src.Activations import Activations as Act

class NeuralNetwork:
    def __init__(self, layers, activation, output_activation, loss, seed, init, 
                 lambda_reg=0.0, dropout_rate=0.0, training=True):
        np.random.seed(seed)
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.init = init
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        self.training = training
        self.params_dict = {}
        self.dropout_masks = []
        self._init_params()

    def _init_params(self):
        for l in range(1, len(self.layers)):
            n_in, n_out = self.layers[l - 1], self.layers[l]

            if self.init == 'he':
                scale = np.sqrt(2.0 / n_in)
            elif self.init == 'xavier':
                scale = np.sqrt(1.0 / n_in)
            else:
                scale = 0.01

            self.params_dict[f"W{l}"] = np.random.randn(n_out, n_in) * scale
            self.params_dict[f"b{l}"] = np.zeros((n_out, 1))

    def forward(self, X):
        A = X.T
        A_list, Z_list = [A], []
        self.dropout_masks = []
        L = len(self.layers) - 1

        for l in range(1, L + 1):
            W, b = self.params_dict[f"W{l}"], self.params_dict[f"b{l}"]
            Z = np.dot(W, A) + b
            if l < L:
                A = getattr(Act, self.activation)(Z)
                if self.training and self.dropout_rate > 0:
                    D = (np.random.rand(*A.shape) < self.dropout_rate).astype(float)
                    A *= D
                    A /= self.dropout_rate
                    self.dropout_masks.append(D)
                else:
                    self.dropout_masks.append(np.ones_like(A))
            else:
                A = getattr(Act, self.output_activation)(Z)

            Z_list.append(Z)
            A_list.append(A)

        return A, {"A_list": A_list, "Z_list": Z_list}

    def backward(self, X, Y, cache):
        A_list = cache["A_list"]
        m = X.shape[0]
        grads = {}
        L = len(self.layers) - 1
        Y_T = Y.T

        A_L = A_list[-1]
        dZ = (A_L - Y_T) / m

        grads[f"W{L}"] = np.dot(dZ, A_list[-2].T) + self.lambda_reg * self.params_dict[f"W{L}"]
        grads[f"b{L}"] = np.sum(dZ, axis=1, keepdims=True)

        for l in range(L - 1, 0, -1):
            W_next = self.params_dict[f"W{l+1}"]
            A_prev = A_list[l - 1]
            A_curr = A_list[l]
            D = self.dropout_masks[l - 1]

            dA = np.dot(W_next.T, dZ)
            if self.dropout_rate > 0:
                dA *= D
                dA /= self.dropout_rate

            dZ = dA * getattr(Act, f"d{self.activation}")(A_curr)

            grads[f"W{l}"] = np.dot(dZ, A_prev.T) + self.lambda_reg * self.params_dict[f"W{l}"]
            grads[f"b{l}"] = np.sum(dZ, axis=1, keepdims=True)

        return grads

    def compute_loss(self, Y_true, Y_pred):
        eps = 1e-8
        if self.loss == 'cce':
            return -np.mean(np.sum(Y_true * np.log(Y_pred.T + eps), axis=1))
        elif self.loss == 'mse':
            return np.mean((Y_true - Y_pred.T) ** 2)

    def params(self):
        return self.params_dict

    def zero_grad(self):
        pass

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False
