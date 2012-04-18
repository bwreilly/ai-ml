# ###Single variate functions #


def update(m1, v1, m2, v2):
    """ Calculate new mean and variance given priors

        This will reduce the variance, increasing convidence of the estimate.
    """
    mean = (v2 * m1 + v1 * m2) / (v1 + v2)
    var = 1 / (1 / v1 + 1 / v2)
    return mean, var


def predict(m1, v1, m2, v2):
    """ Convolution of two gaussian distributions. Overall, this will increase the variance.
    """
    mean = m1 + m2
    var = v1 + v2
    return mean, var

# ###Multivariate
# This is the same operation as above, but in multiple dimensions using numpy matrices.
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
        # (Credit to Sebastian Thrun for the matrix operations)
        I = identity(P.shape)
        for m in measurements:
            # Prediction step. Typically movement.
            x = (self.F * x) + self.u
            P = self.F * P * self.F.transpose()

            # Measurement update.
            Z = matrix(m)
            y = Z.transpose() - (self.H * x)
            S = self.H * P * self.H.transpose() + self.R
            K = P * self.H.transpose() * S.inverse()
            x = x + (K * y)
            P = (I - (K * self.H)) * P
