from old.plotData import plotData as pltd
from supporting_functions import gradientDescent
import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:,0]
x_wb = np.append(np.ones((len(x),1)),x.reshape(-1,1),axis=1)
y_profit = data[:,1]


theta, j_hist = gradientDescent(x_wb, y_profit, np.zeros((2)), 0.01, 1500)

pltd(x=x,
     y=y_profit,
     title='Food truck profit',
     xlabel='Population of City in 10k',
     ylabel='Profit in $10k',
     )

print('Section 2.1 - Plotting the data')
# Get the plot axis to plot the linear regression line and save the figure
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, label='Linear Regression')
plt.legend()
plt.savefig('figures/plot_of_ex1data1_with_regression_line.png')
print('Saved plot_of_ex1data1_with_regression_line.png in directory figures')

# Make the prediction using the calculated theta
print('Profit prediction for 35,000 people: $%f' %((theta[0] + theta[1]*3.5)*10000))
print('Profit prediction for 70,000 people: $%f' % ((theta[0] + theta[1]*7)*10000))
