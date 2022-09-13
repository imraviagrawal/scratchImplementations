# https://github.com/imraviagrawal/machine_learning_basics/blob/master/linear_regression.ipynb

# linear regression with native numpy
# given a input x return a real number y.
# y = wx + b #
# cost function  squared error # 1/2*(y_hat - y)**2
# gradient decent optimization w = w - alpha*dw, dw = -x(y_hat - y), db=-2
import numpy as np

class LinearRegression():
    def __int__(self, n_iter, alpha):
        self.n_iter=n_iter
        self.alpha=alpha

    def fit(self, X, y):
        n, m = X.shape
        jws = []
        self.w= np.random.rand(m, 1)
        self.b = np.random.rand(n, 1)
        for _ in self.n_iter:
            y_hat = self.hypothesis_function(X)
            jw=self.cost_function(y_hat, y)
            jws.append(jw)

            # gradient and optimize
            dJ_dw = (2 / n) * np.dot(X.T, (y_hat-y))
            dJ_db = (2 / n) * np.sum((y_hat - y))

            self.w = self.w + self.alpha*dJ_dw
            self.b = self.b + self.alpha*dJ_db

        return jws


    def predict(self, X):
        return np.dot(X.T, self.w) + self.b

    def cost_function(self, y_hat, y):
        # sum for each sample
        return (1/(2*y_hat.shape[0]))*np.sum((y_hat - y)**2)

    def hypothesis_function(self, X):
        # x(n, m), w(m, 1), b(m, 1)
        return np.dot(X, self.w) + self.b

    def closed_form(self, X, y):
        pass