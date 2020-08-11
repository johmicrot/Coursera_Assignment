import numpy as np

def computeCost(x, y, theta):
    """
    A weighted Mean Square Error Cost function is computed here.  A an extra 1/2 is used to
    cancel the 2 from the square when the darivative is computed.
    :param prediction: pre-calculated prediction
    :param y:  Ground truth, or target
    :param theta: Prediction weights
    :return: Cost function J
    """
    prediction = np.dot(x,theta)
    m = len(y)
    error = prediction - y
    J = np.sum(error**2)/(2*m)

    return(J)
