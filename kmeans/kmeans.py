import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=5, n_iter=100, plot_steps=False):
        self._k = k
        self._n_iter = n_iter
        self._centroids = None
        self._clusters = [[]] * self._k
        self._plot_steps = plot_steps

    def fit(self, X):
        n_samples = X.shape[0]
        # initialize centroids
        centroid_idxs = np.random.choice(n_samples, self._k, replace=False)
        self._centroids = X[centroid_idxs, :]

        for _ in range(self._n_iter):
            # create clusters
            self._clusters = self._create_cluster(X)

            if self._plot_steps:
                self.plot(X)
            
            # compute new centroids
            prev_centroids = self._centroids.copy()
            self._centroids = self._calculate_centroids()

            # break if the algorithm has converged
            if self._has_converged(prev_centroids, self._centroids):
                break

            if self._plot_steps:
                self.plot(X)

    def predict(self, X):
        return [self._predict_helper(x) for x in X]
    
    def _predict_helper(self, x):
        distances = []
        # calculate the distance between the point and the centroids
        for centroid in self._centroids:
            distances.append(self._calculate_distance(centroid, x))

        # return the index of the closest centroids to the point
        return np.argmin(distances).tolist()

    def _create_cluster(self, X):
        clusters = [[] for _ in range(self._k)]

        for x in X:
            distances = []
            # calculate the distance between the point and each centroid
            for centroid in self._centroids:
                distances.append(self._calculate_distance(x, centroid))
            closest_centroid = np.argmin(distances)
            # assign the point to the closest centroid
            clusters[closest_centroid].append(x)

        return clusters

    def _calculate_centroids(self):
        centroids = []
        
        for cluster in self._clusters:
            centroids.append(np.mean(cluster, axis=0))

        return centroids


    def _has_converged(self, old_centroids, new_centroids):
        distances = []
        for i in range(len(old_centroids)):
            distances.append(self._calculate_distance(old_centroids[i], new_centroids[i]))

        return np.sum(distances) == 0

    def _calculate_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def plot(self, X, labels):
        _, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(*X.T, c=labels)

        for point in self._centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        ax.legend()
        plt.show()

        