import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(x):
    """
    Takes in N number of non-normalized features, and returns all features individually normalized
    Performs the same functionality as sklearn.preprocessing.StandardScaler
    Normally using sklearn would be preferred for simplicity.
    :param x: multiple features
    :return:  normalized features individually
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    return (x-mu)/sigma

def computeCost(x, y, theta):
    """
    A weighted Mean Square Error Cost function is computed here. A an extra 1/2 is used to
    cancel the 2 from the square when the darivative is computed.
    :param prediction: pre-calculated prediction
    :param y:  Ground truth, or target
    :param theta: Prediction weights
    :return: Cost function J
    """
    # Simple matrix multiplication of a [a,b] and [b,1] to get a [a,1]
    # vector, or 'a' number of predictions
    prediction = np.dot(x, theta)
    m = len(y)
    error = prediction - y
    cost = np.sum(error**2)/(2*m)

    return cost


def gradientDescent(x, y, theta, alpha, N):
    """
    Perform N steps of gradient descent. Every step takes the whole dataset as a batch
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

        # Theta_0 is the bias
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