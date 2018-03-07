# demonstrates how to calculate the cross-entropy error function
# in numpy.
#
# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python

<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future



>>>>>>> upstream/master
import numpy as np

N = 100
D = 2


X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
<<<<<<< HEAD
    for i in xrange(N):
=======
    for i in range(len(T)):
>>>>>>> upstream/master
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

<<<<<<< HEAD
print cross_entropy(T, Y)
=======
print(cross_entropy(T, Y))
>>>>>>> upstream/master

# try it with our closed-form solution
w = np.array([0, 4, 4])

# calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)

# calculate the cross-entropy error
<<<<<<< HEAD
print cross_entropy(T, Y)
=======
print(cross_entropy(T, Y))
>>>>>>> upstream/master

