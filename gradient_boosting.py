from utils import calculate_rmse, calculate_accuracy

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split


class GradientBoostingRegressor:
    def __init__(
        self, n_estimators=10, max_depth=1, min_samples_split=5, learning_rate=0.01
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._lr = learning_rate

    def fit(self, X, y):

        self._f0 = fm = np.mean(y)
        self._trees = []

        for _ in range(self._n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self._max_depth, min_samples_split=self._min_samples_split
            )

            # calculate the residuals and fit a weak learner
            residuals = y - fm
            tree.fit(X, residuals)
            # predict the residuals and add that to the overall prediction
            fm += self._lr * tree.predict(X)

            self._trees.append(tree)

    def predict(self, X):

        predictions = self._f0 + self._lr * np.sum(
            [estimator.predict(X) for estimator in self._trees], axis=0
        )

        return predictions


class GradientBoostingClassifier:
    def __init__(
        self, n_estimators=10, max_depth=1, min_samples_split=5, learning_rate=0.01
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._lr = learning_rate

    def fit(self, X, y):
        self._trees = []
        labels, counts = np.unique(y, return_counts=True)
        p = np.argmax(labels)
        log_of_odds = np.log(counts[p] / counts[~p])
        self._f0 = fm = np.exp(log_of_odds) / (1 + np.exp(log_of_odds))

        for _ in range(self._n_estimators):
            # calculate residuals
            residuals = y - fm
            # fit the tree for residuals
            tree = DecisionTreeRegressor(
                max_depth=self._max_depth, min_samples_split=self._min_samples_split
            )
            tree.fit(X, residuals)
            # transformation
            reg_pred = tree.predict(X) / (fm * (1 - fm))
            # Add the result to previous tree output
            log_of_odds_prediction = fm + self._lr * reg_pred
            new_prob = np.exp(log_of_odds_prediction) / (
                1 + np.exp(log_of_odds_prediction)
            )
            fm = new_prob

            self._trees.append(tree)

    def predict(self, X):

        predictions = self._f0 + self._lr * np.sum(
            [estimator.predict(X) for estimator in self._trees], axis=0
        )

        return np.where(predictions >= 0.5, 1, 0)


if __name__ == "__main__":
    # classification
    bc = load_breast_cancer()
    gbc = GradientBoostingClassifier(n_estimators=25, max_depth=10)
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=1234
    )
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))

    # regression
    d = load_diabetes()
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=10)
    X, y = d.data, d.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    print("RMSE:", calculate_rmse(y_pred, y_test))
