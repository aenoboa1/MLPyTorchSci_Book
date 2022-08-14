from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np


class lr_torch(object):
    def gen_data(self, X_train, y_train):
        """
        args: X_train, y_train
        Normalize the training set (mean centering and dividing by the standard deviation), create DataLoader
        """
        X_train = np.arange(10, dtype='float32').reshape((10, 1))

        y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6,
                            7.4, 8.0, 9.0], dtype='float32')

        X_train_normalized = (X_train - np.mean(X_train) / np.std(X_train))
        X_train_normalized = torch.from_numpy(
            X_train_normalized)  # Create tensor from numpy
        y_train = torch.from_numpy(y_train)
        train_ds = TensorDataset(X_train_normalized, y_train)
        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        return train_dl

    def loss_fn(self, input, target):
        return(input-target).pow(2).mean()

    def model(self, xb):
        torch.manual_seed(1)
        weights = torch.randn(1)
        weights.requires_grad_()  # Unfreezing method for nn
        bias = torch.zeros(1, requires_grad=True)
        return xb @ weights + bias

    def train(self, X_train, y_train):
        train_dl = self.gen_data(X_train, y_train)
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in train_dl:
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
            with torch.no_grad():
                weights -= weights.grad * self.learning_rate
                bias -= bias.grad * self.learning_rate
                weights.grad.zero_()
                bias.grad.zero_()
            if epoch % self.log_epochs == 0:
                print(f'Epoch {epoch} Loss {loss.item()}')

    def __init__(self, X_train, y_train):
        self.learning_rate = 0.001
        self.num_epochs = 200
        self.log_epochs = 10
        self.batch_size = 1
        self.gen_data(X_train, y_train)
