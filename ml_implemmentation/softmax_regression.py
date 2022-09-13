import numpy as np

class SoftmaxRegression():
    def __init__(self, n_iter, alpha):
        # self.X = x
        # self.y = y
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self,x, y):
        self.X = x
        n_samples, m_features = x.shape
        self.y = y
        # classes
        _, m_classes = y.shape
        jws = []

        # initialize the weight
        self.w = np.random.rand(m_features, m_classes)
        self.b = np.random.rand(n_samples, m_classes) # (n, c)

        for _ in range(self.n_iter):
            y_hat = self.forward(self.X)
            loss = self.cost_function(y_hat, self.y)
            jws.append(loss)

            # calculate the gradient
            dj_dw = (1/n_samples)*np.dot(self.x.T, (y_hat-self.y))  # -x(y_hat-y) (n, m)dot(m, c)
            dj_db = (1/n_samples)*np.sum(y_hat-self.y, axis=1)

            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

        return jws

    def predict(self, X):
        fw = self.forward(X)
        y = self.softmax(fw)
        return np.argmax(y)

    def cost_function(self, y_hat, y):
        n_samples = y_hat.shape[0]
        ## -1/m*(sumi(sumj(ylogYhat))
        j_sum = np.sum(np.sum(y*np.log(y_hat), axis=1), axis=0) # (n, 1)
        # equivalet
        # j_sum = np.sum(y*no.log(y_hat))
        return (-1/n_samples)*j_sum

    def forward(self, X):
        y_hat = np.dot(X, self.w) + self.b # (X (n, c))
        return self.softmax(y_hat)

    def softmax(self, y):
        # softmax
        y_hat_exp = np.exp(y)

        return y_hat_exp/(np.sum(y_hat_exp, axis=1, keepdims=True))