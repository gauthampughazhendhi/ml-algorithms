from utils import calculate_accuracy, unit_step_activation

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self._lr = learning_rate
        self._n_iter = n_iter
        self._activation_func = unit_step_activation

    def fit(self, X, y):
        y = np.where(y > 0, 1, 0)
        self._w, self._b = np.zeros(X.shape[1]), 0

        for _ in range(self._n_iter):
            for i, x in enumerate(X):
                # calculate y_hat based on unit step activation function
                y_hat = self._activation_func(np.dot(x, self._w) + self._b)
                # update the parameters
                self._w += self._lr * (y[i] - y_hat) * x
                self._b += self._lr * (y[i] - y_hat)

    def predict(self, X):
        return self._activation_func(np.dot(X, self._w) + self._b)


if __name__ == "__main__":
    bc = load_breast_cancer()
    p = Perceptron(n_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=1234
    )
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
