from collections import defaultdict
from utils import calculate_accuracy
import numpy as np

from sklearn.model_selection import train_test_split


class MultinomialNB:
    def __init__(self, alpha=1):
        self._alpha = alpha

    def fit(self, X, y):
        n_samples = X.shape[0]
        labels, counts = np.unique(y, return_counts=True)

        self._priors = {}
        self._likelihood = defaultdict(defaultdict)

        for i, label in enumerate(labels):
            # calculate priors
            self._priors[label] = np.log(counts[i] / n_samples)
            idxs = np.argwhere(y == label)
            # for every feature calculate the likelihood for all the unique values
            # belonging to a particular class
            for j, feature in enumerate(X.T):
                values, value_counts = np.unique(feature[idxs], return_counts=True)
                # set default value to handle any missing feature-value pairs
                self._likelihood[label][j] = defaultdict(lambda: np.log(1e-5))
                for k, value in enumerate(values):
                    self._likelihood[label][j][value] = np.log(
                        (value_counts[k] + self._alpha) / len(idxs)
                    )

    def predict(self, X):
        predictions = []
        for x in X:
            # for each sample find label with max posterior probability
            predicted_label = None
            max_p = -float("inf")
            for label in self._priors:
                # compute posterior with prior and likelihood
                prior = self._priors[label]
                likelihood = self._get_likelihood(label, x)
                p = prior + likelihood
                if p > max_p:
                    predicted_label = label
                    max_p = p
            predictions.append(predicted_label)

        return predictions

    def _get_likelihood(self, label, x):
        likelihood = 0
        for i, value in enumerate(x):
            likelihood += self._likelihood[label][i][value]

        return likelihood


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a dataset with continuous features
    X, y = make_classification(
        n_samples=1000,  # Number of samples
        n_features=5,  # Number of features
        n_informative=5,  # Number of informative features
        n_redundant=0,  # No redundant features
        n_clusters_per_class=1,
        random_state=42,
    )

    # Convert continuous features to categorical by binning
    # Here we use 3 bins as an example, which will create 3 categorical levels per feature
    X_binned = np.apply_along_axis(
        lambda x: np.digitize(x, bins=np.linspace(x.min(), x.max(), 4)), axis=0, arr=X
    )

    p = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(
        X_binned, y, test_size=0.2, random_state=1234
    )
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    print("Accuracy:", calculate_accuracy(y_pred, y_test))
