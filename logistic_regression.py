from typing import List, Union

from utils import calculate_accuracy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self._lr = lr
        self._n_iter = n_iter

    def _compute_sigmoid(self, z: np.array) -> np.array:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: List[List[Union[int, float]]], y: List[int]) -> None:
        X, y = np.array(X), np.array(y)
        self._w, self._b = np.zeros(X.shape[1]), 0
        n = X.shape[0]

        for _ in range(self._n_iter):
            # compute y_pred
            y_pred = self._compute_sigmoid(np.dot(self._w, X.T) + self._b)

            # calculate gradients
            dw = -1 / n * np.dot(y - y_pred, X)
            db = -1 / n * np.sum(y - y_pred)

            # update parameters
            self._w -= self._lr * dw
            self._b -= self._lr * db

    def predict(self, X: List[List[Union[int, float]]]) -> List[int]:
        X = np.array(X)
        y_pred = self._compute_sigmoid(np.dot(self._w, X.T) + self._b)

        return [1 if y >= 0.5 else 0 for y in y_pred]


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    log_r = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=1234
    )
    log_r.fit(X_train, y_train)
    y_pred = log_r.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
