import numpy as np
import matplotlib.pyplot as plt

class PerceptronEarlyStopping:
    """
    Implémentation avancée du Perceptron :
    - Early stopping (arrêt anticipé)
    - Shuffle des données à chaque époque
    - Tracking du meilleur score
    """

    def __init__(self, learning_rate=0.01, n_iter=1000, patience=10, shuffle=True, verbose=False):
        """
        learning_rate : taux d'apprentissage
        n_iter : nombre maximum d'époques
        patience : nombre d'époques sans amélioration avant arrêt
        shuffle : mélanger les données à chaque époque
        verbose : afficher les logs
        """
        self.lr = learning_rate
        self.n_iter = n_iter
        self.patience = patience
        self.shuffle = shuffle
        self.verbose = verbose
        self.errors_ = []
        self.weights_ = None
        self.bias_ = 0
        self.best_error_ = np.inf
        self.best_weights_ = None
        self.best_bias_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0

        # Convertir y ∈ {0,1} ➝ y ∈ {-1,+1}
        y_ = np.where(y == 0, -1, 1)

        no_improve_count = 0

        for epoch in range(self.n_iter):
            errors = 0

            # Mélange des données
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X, y_ = X[idx], y_[idx]

            for xi, target in zip(X, y_):
                update = self.lr * (target - self.predict_single(xi))
                if update != 0.0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors += 1

            self.errors_.append(errors)

            # Early stopping
            if errors < self.best_error_:
                self.best_error_ = errors
                self.best_weights_ = self.weights_.copy()
                self.best_bias_ = self.bias_
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.verbose:
                print(f"[Époque {epoch+1}/{self.n_iter}] Erreurs : {errors} (patience={no_improve_count})")

            if no_improve_count >= self.patience:
                if self.verbose:
                    print(f"Arrêt anticipé à l'époque {epoch+1} (aucune amélioration depuis {self.patience} époques).")
                break

        # Restaure les meilleurs poids
        if self.best_weights_ is not None:
            self.weights_ = self.best_weights_
            self.bias_ = self.best_bias_

        return self

    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_

    def predict_single(self, x):
        return 1 if self.net_input(x) >= 0 else -1

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def plot_errors(self):
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(self.errors_)+1), self.errors_, marker='o')
        plt.xlabel('Époques')
        plt.ylabel("Nombre d'erreurs")
        plt.title('Courbe de convergence du Perceptron (avec Early Stopping)')
        plt.grid(True)
        plt.show()

    def get_params(self):
        return self.weights_, self.bias_
