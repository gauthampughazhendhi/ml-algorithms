from typing import List, Union

import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 100):
        self._lr = lr
        self._n_iter = n_iter

    def fit(self, X: List[List[Union[int, float]]], y: List[float]) -> None:
        X, y = np.array(X), np.array(y)
        self._w, self._b = np.zeros(X.shape[1]), 0
        n = X.shape[0]

        for _ in range(self._n_iter):
            # compute gradients
            y_pred = np.dot(self._w, X.T) + self._b
            dw = (-2 / n) * np.dot(y - y_pred, X)
            db = (-2 / n) * np.sum(y - y_pred)

            # update parameters
            self._w = self._w - self._lr * dw
            self._b = self._b - self._lr * db

    def predict(self, X: List[List[Union[int, float]]]) -> List[float]:

        X = np.array(X)
        print(self._w, self._b)
        return (np.dot(self._w, X.T) + self._b).tolist()


if __name__ == "__main__":
    linear_regression = LinearRegression(n_iter=1000)
    linear_regression.fit([[1], [2], [3], [4]], [1, 2, 3, 4])
    print(linear_regression.predict([[6]]))
