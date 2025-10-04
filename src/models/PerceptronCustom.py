# Étape 1 : importer les bonnes bibliothèques
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import matplotlib.pyplot as plt

# Étape 2 : définir la classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000, patience=10, shuffle=True, activation='step', verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.patience = patience
        self.shuffle = shuffle
        self.activation = activation
        self.verbose = verbose
        self.errors_ = []
        self.weights_ = None
        self.bias_ = 0
        self.best_error_ = float('inf')
        self.best_weights_ = None
        self.best_bias_ = None

    def activation_func(self, z):
        if self.activation == 'step':
            return np.where(z >= 0, 1, -1)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Fonction d'activation inconnue : {self.activation}")

    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_

    def predict_single(self, x):
        z = self.net_input(x)
        if self.activation == 'step':
            return 1 if z >= 0 else -1
        elif self.activation == 'sigmoid':
            return 1 if 1 / (1 + np.exp(-z)) >= 0.5 else -1
        elif self.activation == 'tanh':
            return 1 if np.tanh(z) >= 0 else -1
        elif self.activation == 'relu':
            return 1 if np.maximum(0, z) >= 0.5 else -1
        else:
            raise ValueError(f"Fonction d’activation inconnue : {self.activation}")

    def predict(self, X):
        z = self.net_input(X)
        a = self.activation_func(z)
        if self.activation in ['sigmoid', 'tanh', 'relu']:
            return np.where(a >= 0.5, 1, 0)
        else:
            return np.where(a >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        y_ = np.where(y == 0, -1, 1)
        no_improve_count = 0

        for epoch in range(self.n_iter):
            errors = 0
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y_ = X[idx], y_[idx]

            for xi, target in zip(X, y_):
                z = np.dot(xi, self.weights_) + self.bias_
                output = self.activation_func(z)
                update = self.lr * (target - output)
                if update != 0.0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors += 1

            self.errors_.append(errors)

            if errors < self.best_error_:
                self.best_error_ = errors
                self.best_weights_ = self.weights_.copy()
                self.best_bias_ = self.bias_
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                break

        if self.best_weights_ is not None:
            self.weights_ = self.best_weights_
            self.bias_ = self.best_bias_

        return self
