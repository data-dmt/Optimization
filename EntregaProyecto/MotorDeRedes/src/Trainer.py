import numpy as np

class Trainer:
    def __init__(self, network, optimizer, loss_fn, patience=10, decay=0.9):
        self.net = network
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.decay = decay

    def _accuracy(self, Y_true, Y_pred):
        pred = np.argmax(Y_pred, axis=0)
        true = np.argmax(Y_true, axis=1)
        return np.mean(pred == true)

    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=10, batch_size=64, shuffle=True, verbose=True):
        n = X_train.shape[0]
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = np.random.permutation(n)
                X_train, Y_train = X_train[idx], Y_train[idx]
            batch_losses, batch_accs = [], []

            for i in range(0, n, batch_size):
                Xb, Yb = X_train[i:i + batch_size], Y_train[i:i + batch_size]
                Y_hat, cache = self.net.forward(Xb)
                loss = self.loss_fn(Yb, Y_hat)
                grads = self.net.backward(Xb, Yb, cache)
                updated = self.opt.update(self.net.params(), grads)
                self.net.params_dict = updated
                acc = self._accuracy(Yb, Y_hat)
                batch_losses.append(loss)
                batch_accs.append(acc)

            train_loss, train_acc = np.mean(batch_losses), np.mean(batch_accs)
            hist["train_loss"].append(train_loss)
            hist["train_acc"].append(train_acc)

            if X_val is not None and Y_val is not None:
                Y_val_pred, _ = self.net.forward(X_val)
                val_loss = self.loss_fn(Y_val, Y_val_pred)
                val_acc = self._accuracy(Y_val, Y_val_pred)
                hist["val_loss"].append(val_loss)
                hist["val_acc"].append(val_acc)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.patience:
                        print(f"Early stopping: sin mejora tras {self.patience} épocas.")
                        break
            else:
                val_loss, val_acc = None, None

            if epoch % 20 == 0:
                self.opt.lr *= self.decay
                print(f"Decaimiento del learning rate: nuevo lr = {self.opt.lr:.6f}")

            if verbose:
                msg = f"Época {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                print(msg)

        return hist
