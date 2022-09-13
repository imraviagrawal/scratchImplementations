# logistics regression implementation binary classification
# y_hat = sig(wx + b)
# sig = 1/(1+e-z), z=wx+b
# Jw = 1/n*sum((ylogy_hat + (1-y)log(1-y_hat)))
import numpy as np

class LogisticsRegression():
    def __int__(self, n_iter, alpha):
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self, X, y):
        n, m = X.shape
        self.W = np.random.rand(m, 1) # zero or one class
        self.b = np.random.rand(n, 1)
        jws = []

        for _ in range(self.n_iter):
            y_hat = self.sigmoid(X)
            jw = self.loss(y, y_hat)
            jws.append(jw)

            dj_dw = (1 / n) * np.dot(X.T, (y_hat - y))
            dJ_db = (1 / n) * np.sum((y_hat - y))

            # update
            self.W = self.W + self.alpha*(dj_dw)
            self.b = self.b + self.alpha*(dJ_db)

        return sum(jws)/n

    def loss(self, y, y_hat):
        n = y.shape[0]
        return (-1/n)*np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

    def sigmoid(self, X):
        z = np.dot(X, self.W) + self.b # (n, m)*(m, 1) + (n, 1)
        return 1/(1+np.exp(-z))

    def predict(self, X):
        y_hat = self.sigmoid(X)
        return y_hat[:, -1]>0.5
