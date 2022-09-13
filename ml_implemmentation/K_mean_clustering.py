import numpy as np

class KMeansClustering():
    def __init__(self, X, K):
        """

        :param X: Input data
        :param K: clustering
        :return: None
        """
        self.X = X
        self.k = K
        n_samples, m_features = X.shape[0]
        self.cluster = np.random.rand(self.k, m_features)

    def fit(self, n_iter):
        # (n_samples, m_features) - (K, m_features)
        for _ in range(n_iter):
            self.lookup = {i: [] for i in range(self.k)}  # cluster:indexing
            for i, x in enumerate(self.X):
                cluster = np.argmin(self.distance(x, self.cluster))
                self.lookup[cluster].append(i)

            # update cluster
            for c in range(self.k):
                self.cluster[c, :] = np.mean(self.X[self.lookup[c]], axis=0) # (1, m_features)

        return self.cluster

    def distance(self, a, b):
        sub = np.subtract(a, b)
        return np.abs(np.sum(sub, axis=1))










