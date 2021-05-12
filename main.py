import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fmin_bfgs
import math


def readFile(filePath):
    data = loadtxt(filePath, delimiter=",", dtype=np.float64)
    xn = data[:, 0:2]
    yn = data[:, 2]
    return xn, yn


X, y = readFile("ex2data1.txt")
m, n = X.shape
x = np.ones((m, 1 + n), dtype=np.float64)


def plotData(x, y, figure=1):
    # PLOTDATA Plots the data points X and y into a new figure
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix.

    plt.figure(figure)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    # 2D plot, using the option 'k+' for the positive
    # examples and 'ko' for the negative examples.
    y_1 = np.where(y == 1)[0]
    y_0 = np.where(y == 0)[0]
    x_axis_pos = [x[i][0] for i in y_1]
    y_axis_pos = [x[i][1] for i in y_1]
    x_axis_neg = [x[i][0] for i in y_0]
    y_axis_neg = [x[i][1] for i in y_0]
    plt.plot(x_axis_pos, y_axis_pos, '+', 'y', label="Admitted")
    plt.plot(x_axis_neg, y_axis_neg, 'o', 'b', label="Not admitted")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.style.use('fivethirtyeight')


def sigmoid(z):
    g = (1 / (1 + np.exp(-z)))
    return g


#NOTE here in (1-sigmod(...)) instead of 1 i made 1.00001 to avoid problems of making it zero
def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
    #   % Initialize some useful values
    m = len(y)
    thetaLen = theta.shape[0]
    j = (1. / m) * np.sum(-y * np.log(sigmoid(np.dot(X, np.reshape(theta, (thetaLen, 1))))) - (1 - y) * np.log(
        1.00001 - sigmoid(np.dot(X, np.reshape(theta, (thetaLen, 1))))))
    return j


# additional function, it calls fmin_bfgs but with more this done.
def fimunc(x, y, theta, iter):
    return fmin_bfgs(costFunction, x0=theta, maxiter=iter, args=(x, y))


def plotDecisionBoundary(theta, x, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #   intercept.
    #   MxN, N>3 matrix, where the first column is all-ones
    plt.figure(2)
    plotData(x[:, 1:3], y, 2)
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([min(x[:, 1]) - 2, max(x[:, 1]) + 2])
    plot_y = (-1/theta[2])*(theta[0]+theta[1]*plot_x)
    plt.plot(plot_x, plot_y)

def predict(theta, x):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic
    #regression parameters theta
    # p = PREDICT(theta, X) computes the predictions for X using a
    #threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    prob = sigmoid(x.dot(theta))
    for inx,val in enumerate(prob):
        if val >= 0.5:
            prob[inx] =1
        else:
            prob[inx] =0

    return prob
    pass



# Load Data %
# The first two columns contains the exam scores and the third column
# contains the label.
y = np.reshape(y, (len(y), 1))
# ==================== Part 1: Plotting ==================== %
# We start the exercise by first plotting the data to understand the
# the problem we are working with.
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
plotData(X, y)
input("Program paused. Press enter to continue.")

# ============ Part 2: Compute Cost and Gradient ============
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. You neeed to complete the code in
# costFunction.m

m, n = X.shape
x[:, 1:] = X
initial_theta = np.zeros((n + 1, 1), dtype=np.float64)
cost = costFunction(initial_theta, x, y)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
# print('\n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = np.array([-24, 0.2, 0.2], dtype=np.float64)
test_theta = np.reshape(test_theta, (test_theta.shape[0], 1))
cost = costFunction(test_theta, x, y)

print('Cost at test theta: \n', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
# print(' %f', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Program paused. Press enter to continue.')

# ============= Part 3: Optimizing using fminunc  =============
# In this exercise, you will use a built-in function (fminunc) to find the
# optimal parameters theta.
theta = fimunc(x, y, initial_theta, 1500)

# Print theta to screen
print('Cost at theta found by fminunc: %f\n', costFunction(theta, x, y))
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' %f \n', theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta, x, y)

input("Program paused. Press enter to continue.")
#
#After learning the parameters, you'll like to use it to predict the outcomes
# on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and
# score 85 on exam 2 will be admitted.
#Furthermore, you will compute the training and test set accuracies of
# our model.
# Your task is to complete the code in predict.m
# Predict probability for a student with score 45 on exam 1
# and score 85 on exam 2

prob = sigmoid( np.array([1, 45, 85]).dot( theta))
print('For a student with scores 45 and 85, we predict an admission '+'probability of ', prob)
print('Expected value: 0.775 +/- 0.002\n\n')

p = predict(theta, x)

pprob=[]
for inx,val in enumerate(p):
    if val ==y[inx]:
        pprob.append(1)
    else:
        pprob.append(0)
print('Train Accuracy: \n', np.mean(np.array(pprob)) * 100)
print('Expected accuracy (approx): 89.0\n')
plt.show()
