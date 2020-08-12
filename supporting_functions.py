import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(x):
    """
    Takes in N number of non-normalized features, and returns all features individually normalized
    Performs the same functionality as sklearn.preprocessing.StandardScaler.
    :param x: multiple features
    :return:  normalized features individually
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    return (x-mu)/sigma, mu, sigma

def computeCost(x, y, theta):
    """
    Equation used is from 2.2.1 Update Equations
    Predictions are referred to as "h" in the assignment
    A weighted Mean Square Error Cost function is computed here. A an extra 1/2 is used to
    cancel the 2 from the square when the darivative is computed.
    Supports the vectorized form if the number of features are greater then 1.
    :param x: Input data
    :param y:  Ground truth, or target
    :param theta: Parameter weights
    :return: Cost function J
    """
    # Simple matrix multiplication of a [a,b] and [b,1] to get a [a,1]
    # vector, or 'a' number of predictions
    predictions = np.dot(x, theta)
    m = len(y)
    error = predictions - y
    # Using the value 2 below due to the bias column
    if x.shape[1] <= 2:
        cost = np.sum(error**2)/(2*m)
    # Vectorized form
    else:
        cost = np.dot(error.T, error)/(2*m)
    return cost


def gradientDescent(x, y, theta, alpha, N):
    """
    Equation used is from 2.2.1 Update Equations
    Perform N steps of gradient descent. Every update step takes the whole dataset as a batch, and
    slightly shifts the theta parameters.
    Supports single or multiple variable input.
    :param x:  Input data
    :param y:  Targets
    :param theta:  Model weights
    :param alpha:   Learning rate
    :param num_steps:  Number of update steps to perform
    :return: updated theta
    """
    m = len(y)
    J_record = np.zeros(N)

    for step in range(N):
        prediction = np.dot(x, theta)
        error = prediction - y

        # Theta_0 is the bias.  A simple matrix multiplication with np.dot is needed to
        # generate the gradient used to update the parameter theta
        theta -= alpha * np.dot(error, x)/m
        J_record[step] = computeCost(x, y, theta)

    return theta, J_record


def plotData(x, y, title, xlabel, ylabel):
    """
    Function to plot data points x and y using matplotlib.
    :param x: input x axis data
    :param y:  y axis data
    :param title:  title of the plot
    :param xlabel: label to give x axis
    :param ylabel: label to give y axis
    :return: Nothing
    """

    # plt.figure(figsize=(10, 10))
    plt.scatter(x,y, marker='x', color='r', label='Training data')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return