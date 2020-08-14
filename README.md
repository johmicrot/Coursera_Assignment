

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Takehome assignment for FML job interview</h3>

  <p align="center">
    The objective is to solve single and multivariable linear regression problems from the Coursera Machine Learning course
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* About the Project
* File explenations
* Libraries used
* Executing the assignment
* Examining results
  * 2.1 Plotting the Data
  * 2.2 Gradient Descent
  * 3 Linear regression with multiple variables
  * 3.2.1 Selecting learning rates
  * 3.3  Normal Equations

### File explenations

* DockerFile -              Contains docker commands to create docker image
* Linear_regression.py -    Main file that executes the single and multi-variable assignment.  
* supporting_functions.py - For better organization I preferred to put all the functions into one file. Contains the following functions:
  * featureNormalize -      Normalizes data
  * computeCost -           Compute the cost for single and multiple features
  * gradientDescent -       Performs a specified number of gradient descent updates and returns the updated weights
  * plotData -              Uses matplotlib to build a figure
  
  

### Libraries used

* [Python 3](https://github.com/python)
* [Matplotlib](https://github.com/matplotlib/matplotlib) Used only for plotting
* [Numpy](https://github.com/numpy/numpy) Used for loading in csv, matrix multiplication, and performing an inverse matrix.  

The documentation specifies only Python, scikit-learn, and TensorFlow can be used, for the machine learning parts.  Matplotlib is used to only display resuts, and numpy is used to load in csv files efficiently, and to easily perform matrix operations.


## Executing the assignment

First download the github repository, then navigate to the respository on your local computer.

Create a volume with 
`docker volume create my-vol`

View the location or  "Mountpoint" of where you volume saves images with 
`docker volume inspect my-vol`

To create a docker image run 
`docker build -t fml .`  (if your using linux type sudo before all commands)'

Then execute the docker with saving the graphs to the volume location 
`docker run -d --name fml -v my-vol:/figures fml`

You can view the output of the docker by typing 
` sudo docker logs -f fml`




Assignment answers will display on the command line output, and the plot figure
will be saved to the directory figures/ 

## Examining results

### 2.1 Plotting the Data

<img src=figures/ex1data1.png height="400">
Here we can first get an overview of the data.

### 2.2 Gradient Descent

The cost function we will be minimizing is shown below. This is a MSE function with an extra 1/2 is used to cancel the 2 obtained when taking the derivative.  The compute cost function is implemented in supporting_functions.computeCost.  Both the vectorized and non-vectorized forms are implemented.  If the number of features (excluding bias) is greater then one, then the vectorized for is used.

<img src=figures/cost_function.png height="100">

Our dataset has only one feature so we have only one parameter, plus a bias.  The equation is shown below.  The reason numpy is used is to allow us to simply make a prediction by using the command "np.dot(x, theta)"  where x is the input features, and theta are the features.  

<img src=figures/hypothesis_function.png height="75">

The parameter update equation is given below.  The code implementation is given in supporting_functions.gradientDescent. The function returns the parameter after updating N times, and the cost history for all N updates.

<img src=figures/batch_update_equation.png height="80">

### 2.4 Visualizing J(θ)

Below we can see a 3d and 2d representation of the cost function.  The left image give you a more zoomed in 3d view, while the right image shows you a contour figure.
<img src=figures/3d_cost_visualization.png width="1200">



### 3 Linear regression with multiple variables

Feature normalization is performed with supporting_functions.featureNormalize. This helps with scaling and shifting all the features to approximately the same space which will produce gradients of approximately the same magnitude.  If one feature produces gradients many orders of magnitude larger then another, this can dominate the training and lead to either ineffective models or unstable training.


### 3.2.1 Optional (ungraded) exercise: Selecting learning rates

Below we can see the cost vs iterations for varios learning rates. The largest learning rate reached convergence the quickest.  This is due to the simplicity of the problem as seen when Visualizing J(θ).  The topological space of the cost function is very smooth.  If the cost function had steeper curves, or if we are working with more features that create less smooth function, it is possible that a learning rate of 0.3 would cause instable learning.

<img src=figures/ex1data2_multiple_learning_rates.png height="400">


### 3.3  Normal Equations

The closed form solution below is shown to be euqivalent to the weights obtained though gradient descent.

<img src=figures/closed_form_solution.png height="50">

My code outputs 

`
[340412.65957447 109447.79646964  -6578.35485416] Closed form solution weights  
[340412.65957447 109447.79646964  -6578.35485416] Estimated weights from batch gradient descent
`


