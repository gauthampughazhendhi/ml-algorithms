from utils import calculate_accuracy

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class SVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_=0.01, n_iter=100):
        self._lr = learning_rate
        self._lambda = lambda_
        self._n_iter = n_iter

    def fit(self, X, y):
        self._w = np.zeros(X.shape[1])
        self._b = 0

        for _ in range(self._n_iter):
            for i in range(X.shape[0]):
                y_pred = np.dot(X[i], self._w.T) + self._b
                if y[i] * y_pred >= 1:
                    self._w -= self._lr * (2 * self._lambda * self._w)
                else:
                    self._w -= self._lr * (2 * self._lambda * self._w - np.dot(X[i].T, y[i]))
                    self._b -= self._lr * (-y[i])

    def predict(self, X):
        return np.sign(np.dot(X, self._w.T) + self._b)
    

if __name__ == "__main__":
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    y = np.where(y > 0, 1, -1)
    p = SVMClassifier(n_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
