import numpy as np
from src.NeuralNetwork import NeuralNetwork
from src.OptimizerAdam import OptimizerAdam
from src.Trainer import Trainer
from src.Losses import categorical_cross_entropy

class HyperparameterTuner:
    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.results = []

    def search(
        self,
        lr_list=[0.001, 0.005, 0.01, 0.1],
        batch_sizes=[16, 32, 64],
        regs=[0.0001, 0.001],
        hidden_layers=[[8], [12], [6], [4]],
        activations=["relu", "tanh"],
        betas=[(0.9, 0.999, 0.99, 0.95)],
        dropout_rates=[0.0, 0.2, 0.3, 0.5],
        epochs=[100, 150, 200, 400, 600, 800, 1000],
        patience=10,
        decay=[0.9, 0.95, 0.99],
        seed=42
    ):
        config_id = 1
        total = len(lr_list) * len(batch_sizes) * len(regs) * len(hidden_layers) * len(activations) * len(betas)

        for lr in lr_list:
            for batch_size in batch_sizes:
                for reg in regs:
                    for hidden in hidden_layers:
                        for act in activations:
                            for (b1, b2) in betas:
                                for dr in dropout_rates:
                                    print(f"\n [{config_id}/{total}] lr={lr}, batch={batch_size}, λ={reg}, layers={hidden}, act={act}, β=({b1},{b2})")

                                    model = NeuralNetwork(
                                        layers=[self.X_train.shape[1]] + hidden + [self.Y_train.shape[1]],
                                        activation=act,
                                        output_activation="softmax",
                                        loss="cce",
                                        seed=seed,
                                        init="xavier",
                                        lambda_reg=reg
                                    )

                                    optimizer = OptimizerAdam(lr=lr, beta1=b1, beta2=b2)
                                    trainer = Trainer(model, optimizer, categorical_cross_entropy, patience=patience, decay=decay)
                                    hist = trainer.train(
                                        self.X_train, self.Y_train,
                                        self.X_val, self.Y_val,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        verbose=False
                                    )

                                    best_val_acc = max(hist["val_acc"])
                                    best_val_loss = min(hist["val_loss"])
                                    self.results.append({
                                        "id": config_id,
                                        "lr": lr,
                                        "batch_size": batch_size,
                                        "lambda_reg": reg,
                                        "hidden_layers": hidden,
                                        "activation": act,
                                        "beta1": b1,
                                        "beta2": b2,
                                        "dropout_rate": dr,
                                        "best_val_acc": best_val_acc,
                                        "best_val_loss": best_val_loss

                                    })
                                    config_id += 1

        self.results.sort(key=lambda x: (-x["best_val_acc"], x["best_val_loss"]))
        return self.results
