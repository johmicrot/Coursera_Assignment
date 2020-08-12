import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from supporting_functions import computeCost

theta = [-3.63029144,  1.16636235]  # Hard coded, recompute later

data = np.loadtxt('ex1data1.txt', delimiter=',')
x_pop = data[:,0]
x_prop_wb = np.append(np.ones((len(x_pop),1)),x_pop.reshape(-1, 1), axis=1)
y_profit = data[:,1]

# Matplotlib requires a 2d inputs so we use numpy to generate a grid for X and Y
X, Y = np.meshgrid(np.linspace(-10, 10, 100),
                   np.linspace(-1, 4, 100),
                   indexing='xy')

# Here the grid of cost values are generated on a 100x100 grid with using
# the appropriate X and Y values.
Z = np.zeros((100, 100))
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(x=x_prop_wb,
                         y=y_profit,
                         theta=np.array([X[i, j], Y[i, j]])
                         )

fig = plt.figure(figsize=(25,10))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax1.set_zlabel('Cost')
# Change the view to a specific position
ax1.view_init(elev=10, azim=240)

CS = ax2.contour(X, Y, Z, np.logspace(-2, 3, 20), cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax2.scatter(theta[0],theta[1], c='r')
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0 (Bias)$', fontsize=12)
    ax.set_ylabel(r'$\theta_1$ (Weights)', fontsize=12)
plt.show()