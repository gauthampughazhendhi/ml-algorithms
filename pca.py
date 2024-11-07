import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class PCA:
    def __init__(self, n_components):
        self._n_components = n_components
        self._mean = None
        self.pc = None
        self.pc_importance = None

    def fit(self, X):
        # center X
        self._mean = np.mean(X, axis=0)
        X = X - self._mean

        # Calculate the covariance matrix
        cov = np.cov(X.T)

        # compute the eigen values and vectors
        eigen_values, eigen_vectors = np.linalg.eig(cov)

        # sort based on the maximum variance
        idxs = np.argsort(eigen_values)[::-1]

        self.pc = eigen_vectors[idxs][: self._n_components]
        self.pc_importance = eigen_values[idxs]

    def transform(self, X):
        # center based on training data's mean
        X = X - self._mean

        return np.dot(X, self.pc.T)


if __name__ == "__main__":

    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    print("Importance of principle components: ", pca.pc_importance)

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.savefig("./pca.png")
