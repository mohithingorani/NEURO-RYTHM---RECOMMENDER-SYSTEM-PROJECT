import numpy as np


class LinUCB:
    def __init__(self, n_arms, dim, alpha=1.0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros((dim,)) for _ in range(n_arms)]


    def select(self, x):
    # x: context vector (dim,)
        p = []
        for a in range(self.n_arms):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv.dot(self.b[a])
            p_a = theta.dot(x) + self.alpha * np.sqrt(x.dot(Ainv).dot(x))
            p.append(p_a)
        return int(np.argmax(p)), p


def update(self, arm, x, reward):
    self.A[arm] += np.outer(x,x)
    self.b[arm] += reward * x