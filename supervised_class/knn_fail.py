# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
<<<<<<< HEAD
=======
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


>>>>>>> upstream/master
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

def get_data():
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
<<<<<<< HEAD
    for i in xrange(width):
        t = start_t
        for j in xrange(height):
=======
    for i in range(width):
        t = start_t
        for j in range(height):
>>>>>>> upstream/master
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2 # alternate between 0 and 1
        start_t = (start_t + 1) % 2
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()

    # display the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
<<<<<<< HEAD
    print "Train accuracy:", model.score(X, Y)
=======
    print("Train accuracy:", model.score(X, Y))
>>>>>>> upstream/master
