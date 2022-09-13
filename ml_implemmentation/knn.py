import numpy as np

class KNN():
    def __init__(self):
        pass

    def fit(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X):
        indexs = self.distance(X)
        y_pred = self.y[indexs] # indexs
        return (sum(y_pred)/self.k)>0.5 # true if more are Y else N

    # euclidean distance
    def distance(self, X):
        if len(X.shape) ==1:
            X = X.reshape(1, -1)
        # X --> (1, m)
        # self.X --> (n, m)
        distance = np.sqrt(np.sum((self.X-X)**2, axis=1)) # feature axis
        indexs = np.argmin(distance) # returns the index
        return indexs
