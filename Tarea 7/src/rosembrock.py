""" Rosembrock function """

import numpy as np


def function(x: np.array, n: int = 100) -> float:
    """ Compute the evaluation for Extended Rosembrock function with n=100
        Args:
        x: Array of length=n with x's parameters
        n: Rosembrock, n = 100

        Returns:
            Evaluation of f(X)
    """
    ans = 0.0
    for i in range(n-1):
        ans += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return ans


def gradient(x: np.array, n: int = 100) -> np.array:
    """ Compute the gradient evaluation for Extended Rosembrock function with n=2
        Args:
        x: Array of length=n with x's parameters
        n: Rosembrock, n = 100

        Returns:
            Gradient of f(x1, ..., xn), array with lenght=n
    """
    # grad = np.zeros(n, dtype=np.float64)
    # for i in range(n-1):
    #     grad[i] = -400 * x[i+1] * x[i] + 400 * x[i]**3 + 2 * x[i] -2
    # grad[n-1] = 200 * (x[n-1] - x[n-2]**2)
    # return grad
    grad = np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0])])

    for i in range(1, n-1):
        grad = np.append(grad, [200*(x[i]-x[i-1]**2)-400*(x[i+1]-x[i]**2)*x[i]-2*(1-x[i])])

    grad = np.append(grad, [200*(x[99] - x[98]**2)])

    return grad


def hessian(x: np.array, n: int = 100) -> np.array:
    """ Compute the Hessian evaluation for Extended Rosembrock function with n=2
        Args:
        x: Array of length=n with x's parameters

        Returns:
            Hessian of f(x1, ..., xn), Matrix with size=nxn
    """
    hess = np.zeros((n, n), dtype=np.float64)
    for i in range(n-1):
        hess[i][i] = -400 * x[i+1] + 1200 * x[i]**2 + 2
        hess[i][i] += 200 if i != 0 else 0
        hess[i][i+1] = hess[i+1][i] = -400 * x[i]
    hess[n-1][n-1] = 200.0
    return hess
