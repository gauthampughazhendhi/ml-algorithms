from typing import Any
from dataclasses import dataclass

from utils import calculate_accuracy

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets


@dataclass
class Node:
    feature: int = None
    threshold: Any = None
    left: Any = None
    right: Any = None
    value: Any = None

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=100,
        min_samples_split=2,
        n_features_per_split=None,
    ):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._n_features_per_split = n_features_per_split
        self._root = None

    def fit(self, X, y):
        self._root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if (
            depth >= self._max_depth
            or n_labels == 1
            or n_samples < self._min_samples_split
        ):
            return Node(value=self._get_most_common_label(y))

        # get best split
        feat_idxs = np.random.choice(
            n_features,
            (
                n_features
                if not self._n_features_per_split
                else min(n_features, self._n_features_per_split)
            ),
            replace=False,
        )
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        # split samples
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, features):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in features:
            values = X[:, feature]
            thresholds = np.unique(values)

            for t in thresholds:
                curr_gain = self._get_information_gain(y, values, t)

                if curr_gain > best_gain:
                    best_gain = curr_gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def _get_information_gain(self, y, feature_values, threshold):
        # calculate parent entropy
        p_entropy = self._get_entropy(y)

        # split the parent
        left_idxs, right_idxs = self._split(feature_values, threshold)
        n, n_left, n_right = len(y), len(left_idxs), len(right_idxs)

        # return 0 if the parent cannot be split
        if n_left == 0 or n_right == 0:
            return 0

        # calculate children entropy
        lc_entropy = self._get_entropy(y[left_idxs])
        rc_entropy = self._get_entropy(y[right_idxs])
        c_entropy = (n_left / n) * lc_entropy + (n_right / n) * rc_entropy

        return p_entropy - c_entropy

    def _split(self, X_column, threshold):

        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def predict(self, X):
        predictions = []

        for x in X:
            predictions.append(self._traverse_tree(x, self._root))

        return predictions

    def _get_entropy(self, y):

        hist = np.bincount(y)
        probs = hist / len(y)

        return -np.sum([p * np.log(p) for p in probs if p > 0])

    def _get_most_common_label(self, y) -> int:

        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _traverse_tree(self, x, node):

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    tree = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=1234
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
