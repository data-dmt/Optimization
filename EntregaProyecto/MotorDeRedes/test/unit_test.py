import unittest
import numpy as np
from src.NeuralNetwork import NeuralNetwork
from src.Losses import categorical_cross_entropy, mean_squared_error
from src.Utils import one_hot_encode, normalize, split_data
from src.OptimizerAdam import OptimizerAdam

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        X = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])
        Y = np.array([[0, 1, 1, 0]])
        self.X = X
        self.Y = Y
        self.model = NeuralNetwork(n_input=2, n_hidden=3, n_output=1, activation='sigmoid')
        self.optimizer = OptimizerAdam(lr=0.01)

    def test_forward_output_shape(self):
        Y_hat, _ = self.model.forward(self.X)
        self.assertEqual(Y_hat.shape, (1, self.X.shape[1]))

    def test_loss_mse(self):
        Y_hat, _ = self.model.forward(self.X)
        loss = mean_squared_error(self.Y, Y_hat)
        self.assertGreaterEqual(loss, 0.0)

    def test_loss_cce(self):
        Y = np.eye(3)[np.array([0, 1, 2])]  # one-hot dummy
        Y_hat = np.array([[0.9, 0.1, 0.2],
                          [0.05, 0.8, 0.1],
                          [0.05, 0.1, 0.7]])
        loss = categorical_cross_entropy(Y, Y_hat)
        self.assertGreaterEqual(loss, 0.0)

    def test_training_reduces_loss(self):
        X_train, Y_train = self.X, self.Y
        losses = []
        for _ in range(5):
            Y_hat, cache = self.model.forward(X_train)
            loss = mean_squared_error(Y_train, Y_hat)
            grads = self.model.backward(X_train, Y_train, cache)
            self.model.params = self.optimizer.step(self.model.params, grads)
            losses.append(loss)
        self.assertTrue(losses[-1] < losses[0], "La pérdida no disminuye tras entrenamiento.")


if __name__ == "__main__":
    unittest.main()
