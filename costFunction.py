import numpy as np

def cost_function(prediction, y):
    """
    A weighted Mean Square Error Cost function is computed here.  A an extra 1/2 is used to
    cancel the 2 from the square when the darivative is computed.
    :param x: Input data
    :param y:  Target we want to estimate
    :param theta:   Weights to approximate the target
    :return: Cost function J
    """

    m = len(y)
    # prediction = np.dot(x, theta)
    error = prediction - y

    J = np.sum(error**2)/(2*m)

    return(J)
