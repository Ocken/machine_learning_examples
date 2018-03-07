# shows how linear regression analysis can be applied to polynomial data
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python
<<<<<<< HEAD

=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
>>>>>>> upstream/master

import numpy as np
import matplotlib.pyplot as plt


# load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x]) # add the bias term x0 = 1
<<<<<<< HEAD
=======
    # our model is therefore y_hat = w0 + w1 * x + w2 * x**2
>>>>>>> upstream/master
    Y.append(float(y))

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

<<<<<<< HEAD

# let's plot the data to see what it looks like
plt.scatter(X[:,1], Y)
=======
# let's plot the data to see what it looks like
plt.scatter(X[:,1], Y)
plt.title("The data we're trying to fit")
>>>>>>> upstream/master
plt.show()


# apply the equations we learned to calculate a and b
# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
<<<<<<< HEAD
Yhat = np.dot(X, w)
=======
>>>>>>> upstream/master


# let's plot everything together to make sure it worked
plt.scatter(X[:,1], Y)
<<<<<<< HEAD
plt.plot(sorted(X[:,1]), sorted(Yhat))
# note: shortcut since monotonically increasing
#       x-axis values have to be in order since the points
#       are joined from one element to the next
=======

# to plot our quadratic model predictions, let's
# create a line of x's and calculate the predicted y's
x_line = np.linspace(X[:,1].min(), X[:,1].max())
y_line = w[0] + w[1] * x_line + w[2] * x_line * x_line
plt.plot(x_line, y_line)
plt.title("Our fitted quadratic")
>>>>>>> upstream/master
plt.show()


# determine how good the model is by computing the r-squared
<<<<<<< HEAD
=======
Yhat = X.dot(w)
>>>>>>> upstream/master
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
