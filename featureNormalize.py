import numpy as np

def featureNormalize(x):
    """
    Performs the same functionality as sklearn.preprocessing.StandardScaler
    Normally using sklearn would be preferred for simplicity.
    :param x: multiple features
    :return:  normalized features individually
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x-mu)/sigma
