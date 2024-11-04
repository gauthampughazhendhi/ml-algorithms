from dataclasses import dataclass
from typing import Any

from decision_tree import DecisionTreeClassifier

from utils import calculate_accuracy

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets


@dataclass
class WeakLearner:
    model: Any
    alpha: float


# Binary Class Implementation
class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._weak_learners = []
        self._n_labels = None

    def fit(self, X, y):
        self._n_labels = len(np.unique(y))
        n_samples = X.shape[0]

        w = np.ones(n_samples) / n_samples

        for _ in range(self._n_estimators):
            np.random.seed(1)
            # sample data based on sample weights
            idxs = np.random.choice(n_samples, n_samples, p=w)
            X_sampled, y_sampled = X[idxs, :], y[idxs]

            # fit a weak estimator and get predictions
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X_sampled, y_sampled)
            y_pred = tree.predict(X_sampled)

            # calculate error
            error = np.sum(w * (y_pred != y_sampled))

            # calculate amount of say
            alpha = self._learning_rate * 0.5 * np.log((1 - error) / (error + 1e-8))
            self._weak_learners.append(WeakLearner(tree, alpha))

            # update weights
            sign = np.where(y_sampled == y_pred, 1, -1)
            w *= np.exp(-alpha * sign)
            w /= np.sum(w)

    def predict(self, X):

        predictions = np.zeros(X.shape[0])

        # get predictions from all the weak estimations
        for i, estimator in enumerate(self._weak_learners):
            y_pred = np.array(estimator.model.predict(X))
            predictions[:] += estimator.alpha * np.where(y_pred == 0, -1, 1)

        # return prediction labels based on the sign
        return np.where(predictions > 0, 1, 0)


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    tree = AdaBoostClassifier(n_estimators=10)
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
