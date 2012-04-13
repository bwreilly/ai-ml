# single variate functions #


def update(m1, v1, m2, v2):
    """ Calculate new mean and variance given priors
    """
    mean = (v2 * m1 + v1 * m2) / (v1 + v2)  # bayes rule
    var = 1 / (1 / v1 + 1 / v2)
    return mean, var


def predict(m1, v1, m2, v2):
    """ Convolution of two gaussian distributions
    """
    mean = m1 + m2
    var = v1 + v2
    return mean, var

# multivariate
from numpy import matrix, identity


class Kalman(object):
    """ Kalman filter implementation, see https://en.wikipedia.org/wiki/Kalman_filter """
    def __init__(self, dt, F, H, R, u):
        """
        dt = delta time
        F = next state function
        H = measurement function
        R = measurement uncertainty
        u = external motion

        examples:
        dt = 0.1
        P = matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1000., 0], [0, 0, 0, 1000.]])
        F = matrix([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        H = matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = matrix([[dt, 0], [0, dt]])
        u = matrix([[0.], [0.], [0.], [0.]])
        """
        self.dt = dt
        self.F = F
        self.H = H
        self.R = R
        self.u = u

    def filter(self, x, P, measurements):
        """
        x = initial state (location and velocity)
        P = initial uncertainty
        """
        # credit to Sebastian Thrun for the matrix operations
        I = identity(P.shape)
        for m in measurements:
            # prediction
            x = (self.F * x) + self.u
            P = self.F * P * self.F.transpose()

            # measurement update
            Z = matrix(m)
            y = Z.transpose() - (self.H * x)
            S = self.H * P * self.H.transpose() + self.R
            K = P * self.H.transpose() * S.inverse()
            x = x + (K * y)
            P = (I - (K * self.H)) * P
