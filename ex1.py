from plotData import plotData as pltd
from gradientDescent import gradientDescent as gd
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
sns.set()

data = np.loadtxt('ex1data1.txt', delimiter=',')
x_pop = data[:,0]
y_profit = data[:,1]


features, jhist = gd(x_pop, y_profit, np.zeros((2)), 0.01, 1500)

pltd(x=x_pop,
     y=y_profit,
     title='Food truck profit in a city',
     xlabel='Population of City in 10k',
     ylabel='Profit in $10k',
     )

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = features[0] + features[1] * x_vals
plt.plot(x_vals, y_vals, '--', label='Linear Regression')
plt.legend()
plt.show()