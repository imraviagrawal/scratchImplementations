import numpy as np

class SVR():
    def __init__(self, alpha, C, n_iters, eps):
        """
        You fit error_rate as part of the support vector optimization. requires clarification and changes
        :param alpha:
        :param C:
        :param n_iters:
        :param eps:
        """
        self.alpha = alpha
        self.C = C
        self.n_iters = n_iters
        self.eps = eps

    def fit(self, X, y):
        n_samples, m_features = X.shape
        c_classes = y.shape

        jws = []
        # initialize the weights
        self.w = np.random.rand(m_features, c_classes)
        self.b = np.random.rand(n_samples, c_classes)

        for _ in range(self.n_iters):
            y_hat = self.forward(X)

            # compute_loss
            _loss = self.compute_loss(y_hat, y)
            jws.append(_loss)

            # backpropogate
            dj_dw, dj_db = self.backprop(X, y_hat, y)
            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

        return jws

    def predict(self, X):
        y_hat = self.forward(X)
        return np.any(y_hat>0.5)

    def compute_loss(self, y_hat, y):
        # j(w) = (1/n)*(W**2) + c*(1/N)*sum(max(0, y-(y_hat + eps))
        n_features = y_hat.shape[0]
        distance = np.maximum(0, y + y_hat - self.eps)
        hinge= (self.C/n_features)*np.sum(distance, axis=1)

        # margin
        margin = (1/n_features)*np.dot(self.W.T, self.W)

        total_loss = margin + hinge
        return total_loss

    def forward(self, X):
        y_hat = np.dot(X, self.w) + self.b
        return y_hat

    def backprop(self, X, y_hat, y):
        n_features = y_hat.shape[0]
        # if 1-y*y_hat < 0 then w else w - c*y*x
        distance = y + y_hat
        index = np.where(np.any(distance>self.eps, axis=1))

        base_w = self.W
        # sum the loss for others
        # for greater than zero onces
        ext_w = -self.C*(1/n_features)*np.dot(X[index].T, y[index]) #  (n, m)*(n, c) --> (n, C)


        dj_dw = base_w + ext_w
        dj_db = -self.C * (1 / n_features) * (np.sum(y[index], axis=0))

        return dj_dw, dj_db