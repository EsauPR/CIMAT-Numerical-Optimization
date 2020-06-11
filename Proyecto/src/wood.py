""" Wood function """

import numpy as np


def function(x: np.array) -> float:
    """ Compute the evaluation for Wood Function function with n=100
        Args:
        x: Array of length=4 with x's parameters

        Returns:
            Evaluation of f(X)
    """
    f1 = 100*(x[0]**2 - x[1])**2 + (x[0]-1)**2 + 10.1*(x[1]-1)**2 + (x[2]-1)**2
    f2 = 10.1*(x[3]-1)**2 + 90*(x[2]**2-x[3])**2 +19.8*(x[1]-1)*(x[3]-1)
    return f1 + f2
    # ans = 100 * (x[0]**2 - x[1]) + (x[0] - 1)**2 + (x[2] - 1)**2 + 90 * (x[2]**2 - x[3])**2
    # ans += 10.1 * ((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)
    # return ans


def gradient(x: np.array) -> np.array:
    """ Compute the gradient evaluation for Extended Rosembrock function with n=2
        Args:
        x: Array of length=4 with x's parameters

        Returns:
            Gradient of f(x1, x2, x3, x4), array with lenght=4
    """
    grad = np.zeros(4, dtype=np.float64)
    grad[0] = 400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2
    grad[1] = -200 * (x[0]**2 - x[1]) + 20.2 * (x[1] - 1) + 19.8 * x[3] - 19.8
    grad[2] = 2 * x[2] - 2 + 360 * x[2]**3 - 360 * x[3] * x[2]
    grad[3] = -180 * (x[2]**2 - x[3]) + 20.2 * (x[3] - 1) + 19.8 * (x[1] -1)
    return grad


def hessian(x: np.array) -> np.array:
    """ Compute the Hessian evaluation for Extended Rosembrock function with n=2
        Args:
        x: Array of length=4 with x's parameters

        Returns:
            Hessian of f(x1, x2, x3, x4), Matrix with size=4x4
    """
    hess = np.zeros((4, 4), dtype=np.float64)

    hess[0][0] = 1200 * x[0]**2 - 400 * x[1] + 2
    hess[0][1] = hess[1][0] = -400 * x[0]
    hess[1][1] = 220.2
    hess[2][2] = 1080 * x[2]**2 - 360 * x[3] + 2
    hess[3][1] = hess[1][3] = 19.8
    hess[3][2] = hess[2][3] = -360 * x[2]
    hess[3][3] = 200.2

    return hess
