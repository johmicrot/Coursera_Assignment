import numpy as np
from computeCost import computeCost as cf

def gradientDescent(x, y, theta, alpha, N):
    """
    Perform N steps of gradient descent.
    :param x:  Input data
    :param y:  Targets
    :param theta:  Model weights
    :param alpha:   Learning rate
    :param num_steps:  Number of update steps to perform
    :return: updated theta
    """
    m = len(y)

    J_record = np.zeros(N)

    # Here I will prepend a column of ones to the input features so theta has a
    # placeholder to multiply its bias element.

    for step in range(N):
        prediction = np.dot(x,theta)
        error = prediction - y

        # Here we include an extra 1/2 to
        # theta[0] -= alpha * np.mean(error * x[:,0])
        theta -= alpha * np.dot(error, x)/m
        J_record[step] = cf(x, y, theta)

    return theta, J_record
