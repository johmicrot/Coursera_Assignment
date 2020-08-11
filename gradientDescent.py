import numpy as np
from costFunction import computeCost as cf

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
    x_wb = np.append(np.ones((len(x),1)),x.reshape(-1,1),axis=1)

    bias = 0
    t = 0
    for step in range(N):
        prediction = np.dot(x_wb,theta)
        error = prediction - y

        # Here we include an extra 1/2 to
        theta[0] -= alpha * np.mean(error * x_wb[:,0])
        theta[1] -= alpha * np.mean(error *  x_wb[:,1])
        J_record[step] = cf(x_wb, y, theta)

        # print(theta)
    return theta, J_record
