import numpy as np

"""
loss fucntion 


"""

class SVM():
    def __init__(self, n_inter, C, alpha):
        """

        :param n_inter: epochs
        :param C: hyper parameters
        """

        self.n_inter = n_inter
        self.C = C
        self.alpha = alpha

    def fit(self, X, y):
        # binary class
        n_samples, m_features = X.shape
        jws=[]
        self.W = np.random.rand(m_features, 1) # features, classes
        self.b = np.random.rand(n_samples, 1) # bias

        for _ in range(self.n_inter):
            y_hat = np.dot(X, self.W) + self.b
            _loss=self.cost_function(y_hat, y)
            jws.append(_loss)

            # gradients
            dj_dw, dj_db = self.backProp(y_hat, y, X)
            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

        return jws

    def backProp(self, y_hat, y, X):
        # if distance < 0:
        n_features = y_hat.shape[0]
        distance = 1-y*y_hat #some greater some smaller
        index = np.where(np.any(distance>0, axis=1)) # for smaller

        # add W
        base_loss = self.w # m, c
        base_loss -= (self.C/n_features)*np.dot(X[index].T, y[index]) # (m, n)*(n, c) --> (m, c)

        # bias
        bias_loss = -(self.C/n_features)*np.sum(y[index], axis=0)
        return base_loss, bias_loss

    def cost_function(self, y_hat, y):

        # hinge loss # (m_features, classes)
        #j(w) = 1 / 2 * mod(W)**2 + c*1/N * max(0, 1 - y * f(x))
        distance = 1-y*y_hat
        n_samples = y_hat.shape[0]
        zeros = np.zeros((1, y_hat.shape[0]))
        hinge = ((self.C*1)/n_samples)*np.sum(np.maximum(zeros, distance))

        # mod w
        margin = (1/n_samples)*np.dot(self.W.T, self.W)
        total_loss = margin + hinge
        return total_loss

    def activation_function(self):
        pass

    def predict(self, X):
        y_hat = np.dot(X, self.W) + self.b
        return np.argmax(y_hat, axis=1) # True False