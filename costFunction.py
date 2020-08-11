import numpy as np

def cost_function(input, target, theta):
    m = len(input)
    prediction = np.dot(theta, input)
    error = prediction - target
    J = np.sum(error**2)/(2*m)

    return(J)
