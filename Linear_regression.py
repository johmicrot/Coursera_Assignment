from supporting_functions import *
import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:,0]
x_wb = np.append(np.ones((len(x),1)),x.reshape(-1,1),axis=1)
y_profit = data[:,1]


print('2.1 - Plotting the data')
plotData(x=x,
     y=y_profit,
     title='Food truck profit',
     xlabel='Population of City in 10,000s',
     ylabel='Profit in $10,000s',
     )
plt.savefig('figures/ex1data1.png')
print('\tSaved ex1data1.png in directory figures')

print('2.2.3 -  Computing the cost')

print('\tCost when theta is initalized to zero is: %f' %(computeCost(x_wb,y_profit,np.zeros(2))))

print('2.2.4 - Gradient descent')
theta, j_hist = gradientDescent(x_wb, y_profit, np.zeros((2)), 0.01, 1500)
print('\tTrained ex1data1 with gradient descent, and got h(x) =  %.4f + %.4fx' %(theta[0],theta[1]))

plotData(x=x,
     y=y_profit,
     title='Food truck profit',
     xlabel='Population of City in 10,000s',
     ylabel='Profit in $10,000s',
     )

# Get the plot axis to plot the linear regression line and save the figure
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, label='Linear Regression')
plt.legend()


plt.savefig('figures/ex1data1_with_regression_line.png')
print('\tSaved ex1data1_with_regression_line.png in directory figures')

# Make the prediction using the calculated theta
print('\tProfit prediction for 35,000 people: $%f' %((theta[0] + theta[1]*3.5)*10000))
print('\tProfit prediction for 70,000 people: $%f' % ((theta[0] + theta[1]*7)*10000))



print('3 Linear regression with multiple variables')

data = np.loadtxt('ex1data2.txt', delimiter=',')
# All columns but last are the features
x = data[:,:-1]
y_profit = data[:,-1]
print('3.1 Feature Normalization')
x, mu, sigma = featureNormalize(x)
print('\t Performed normalization using featureNormalize function')
print('\t New mean and std are respectively: %.4f,%.4f' % (np.mean(x), np.std(x)))

num_features = x.shape[1]
print('\t ex1data2 contains %i features' % num_features)
# x.reshape(-1,num_features) is needed in the case that num_features == 1
x_wb = np.append(np.ones((len(x), 1)), x.reshape(-1,num_features), axis=1)


print('3.2.1 Optional (ungraded) exercise: Selecting learning rates')
for alpha in [0.01, 0.03, 0.1, 0.3]:
    weights, jhist = gradientDescent(x_wb, y_profit, np.zeros((3)), alpha, 30)
    plt.plot(jhist, label=alpha)

plt.legend(title='Learning Rates')
plt.xlabel('Training epochs')
plt.ylabel('Cost')
plt.title('Comparing different learning rates')
plt.savefig('figures/ex1data2_multiple_learning_rates.png')
print('\tSaved ex1data2_multiple_learning_rates.png in directory figures')

print('\tBest learning rate from my selection is 0.3')
test_normalized = (np.array([1650,3]) - mu)/sigma
pred = weights[0] + np.dot(weights[1:], test_normalized)
print('\tPrediction onusing weights is : %.2f' % pred)

print('3.3 Normal Equations')
# Left side of closed form solution
left = np.linalg.inv(np.dot(x_wb.T,x_wb))
# Right side of closed form solution
right = np.dot(x_wb.T,y_profit)
# Complete solution
theta = np.dot(left,right)
print('\t%s'%theta, 'Closed form solution weights')

# Here I re-run with more epochs to get a result the same as the closed form solution
approx_weights, jhist = gradientDescent(x_wb, y_profit, np.zeros((3)), 0.3, 500)
print('\t%s'%approx_weights, 'Estimated weights from batch gradient descent')



