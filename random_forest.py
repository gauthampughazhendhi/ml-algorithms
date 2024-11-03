import numpy as np

from decision_tree import DecisionTree
from utils import calculate_accuracy

from sklearn.model_selection import train_test_split
from sklearn import datasets


class RandomForest:
    def __init__(
        self,
        n_trees=10,
        max_depth: int = 100,
        min_samples_split: int = 2,
        n_features_per_split: int = None,
    ):
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._n_features_per_split = n_features_per_split
        self._root = None

    def fit(self, X, y):
        self._trees = []
        # create trees
        for _ in range(self._n_trees):
            tree = DecisionTree(
                self._max_depth, self._min_samples_split, self._n_features_per_split
            )
            # create random subset of data
            X_subset, y_subset = self._get_random_subset(X, y)
            # fit the tree on the random subset
            tree.fit(X_subset, y_subset)
            self._trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self._trees:
            predictions.append(tree.predict(X))
        # swap axes to find the most common label across trees
        predictions = np.swapaxes(predictions, 0, 1)
        return [self._get_most_common_labels(prediction) for prediction in predictions]

    def _get_most_common_labels(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _get_random_subset(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs, :], y[idxs]


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    rf = RandomForest(n_trees=20, n_features_per_split=5)
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=1234
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
