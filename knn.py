from typing import List, Any, Union
import numpy as np


class KNN:
    def __init__(self, k: int) -> None:
        self._k = k

    def fit(self, X: List[List[Any]], y: List[Any]) -> None:
        self._X = np.array(X)
        self._y = np.array(y)

    def _calculate_distance(self, x1: List[Any], x2: List[Any]):
        return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X: List[List[Any]]) -> List[int]:
        return [self._predict_helper(x) for x in X]

    def _predict_helper(self, x: List[Any]) -> int:
        x1 = np.array(x)

        # calculate distance
        distances = [self._calculate_distance(x1, x2) for x2 in self._X]

        # get neighbors
        neighbors = np.argsort(distances)[: self._k]

        # get labels of neighbors
        neighbor_labels = self._y[neighbors]

        # return the label of most common class among neighbors
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        return labels[np.argmax(counts)].tolist()


class KNNRegressor(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X: List[List[Any]]) -> List[Union[int, float]]:
        return [self._predict_helper(x) for x in X]

    def _predict_helper(self, x: List[Any]) -> Union[int, float]:
        x1 = np.array(x)

        # calculate distance
        distances = [self._calculate_distance(x1, x2) for x2 in self._X]

        # get closest neighbors
        neighbors = np.argsort(distances)[: self._k]

        # get values of neighbors
        neighbor_values = self._y[neighbors]

        # return the average value of neighbors as result
        return np.mean(neighbor_values).tolist()


if __name__ == "__main__":
    knn_classifier = KNNClassifier(k=3)
    knn_classifier.fit([[1, 1], [1, 2], [2, 3], [4, 4]], [2, 2, 1, 1])
    print("Predicted Labels:", knn_classifier.predict([[0, 0], [3, 3]]))

    knn_regressor = KNNRegressor(k=2)
    knn_regressor.fit([[1, 1], [1, 2], [2, 3], [4, 4]], [10, 20, 30, 40])
    print("Predicted Values:", knn_regressor.predict([[0, 0], [3, 3]]))
