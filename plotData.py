import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# # Here the pandas library is used to cleanly read in the data
# data = pd.read_csv('ex1data1.txt', header=None, names=['Pop', 'Profit'])


def plotData(x, y, title, xlabel, ylabel):
    """
    :param x: input x axis data
    :param y:  y axis data
    :param title:  title of the plot
    :param xlabel: label to give x axis
    :param ylabel: label to give y axis
    :return: Nothing

    Function to plot data points x and y using seaborn.

    """

    # plt.figure(figsize=(10, 10))
    sns.scatterplot(x,y, marker='x', color='r', label='Training data')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.show()
    return


# plotData(x=data.Pop,
#          y=data.Profit,
#          title='Food truck profit in a city',
#          xlabel='Population of City in 10k',
#          ylabel='Profit in $10k',
#          )