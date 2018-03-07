# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()

<<<<<<< HEAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import plot_k_means, get_simple_data
from datetime import datetime

def get_data(limit=None):
    print "Reading in and transforming data..."
=======
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .kmeans import plot_k_means, get_simple_data
from datetime import datetime

def get_data(limit=None):
    print("Reading in and transforming data...")
>>>>>>> upstream/master
    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


<<<<<<< HEAD
=======
# hard labels
def purity2(Y, R):
    # maximum purity is 1, higher is better
    C = np.argmax(R, axis=1) # cluster assignments

    N = len(Y) # number of data pts
    K = len(set(Y)) # number of labels

    total = 0.0
    for k in range(K):
        max_intersection = 0
        for j in range(K):
            intersection = ((C == k) & (Y == j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N


>>>>>>> upstream/master
def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
<<<<<<< HEAD
    for k in xrange(K):
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0
        for j in xrange(K):
=======
    for k in range(K):
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0
        for j in range(K):
>>>>>>> upstream/master
            intersection = R[Y==j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N


<<<<<<< HEAD
def DBI(X, M, R):
    # lower is better
    # N, D = X.shape
    # _, K = R.shape
    K, D = M.shape

    # get sigmas first
    sigma = np.zeros(K)
    for k in xrange(K):
        diffs = X - M[k] # should be NxD
        # assert(len(diffs.shape) == 2 and diffs.shape[1] == D)
        squared_distances = (diffs * diffs).sum(axis=1)
        # assert(len(squared_distances.shape) == 1 and len(squared_distances) != D)
        weighted_squared_distances = R[:,k]*squared_distances
        sigma[k] = np.sqrt(weighted_squared_distances).mean()

    # calculate Davies-Bouldin Index
    dbi = 0
    for k in xrange(K):
        max_ratio = 0
        for j in xrange(K):
=======
# hard labels
def DBI2(X, R):
    N, D = X.shape
    _, K = R.shape

    # get sigmas, means first
    sigma = np.zeros(K)
    M = np.zeros((K, D))
    assignments = np.argmax(R, axis=1)
    for k in range(K):
        Xk = X[assignments == k]
        M[k] = Xk.mean(axis=0)
        # assert(Xk.mean(axis=0).shape == (D,))
        n = len(Xk)
        diffs = Xk - M[k]
        sq_diffs = diffs * diffs
        sigma[k] = np.sqrt( sq_diffs.sum() / n )


    # calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K



def DBI(X, M, R):
    # ratio between sum of std deviations between 2 clusters / distance between cluster means
    # lower is better
    N, D = X.shape
    K, _ = M.shape

    # get sigmas first
    sigma = np.zeros(K)
    for k in range(K):
        diffs = X - M[k] # should be NxD
        squared_distances = (diffs * diffs).sum(axis=1) # now just N
        weighted_squared_distances = R[:,k]*squared_distances
        sigma[k] = np.sqrt( weighted_squared_distances.sum() / R[:,k].sum() )

    # calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
>>>>>>> upstream/master
            if k != j:
                numerator = sigma[k] + sigma[j]
                denominator = np.linalg.norm(M[k] - M[j])
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K


def main():
<<<<<<< HEAD
=======
    # mnist data
>>>>>>> upstream/master
    X, Y = get_data(10000)

    # simple data
    # X = get_simple_data()
    # Y = np.array([0]*300 + [1]*300 + [2]*300)

<<<<<<< HEAD
    print "Number of data points:", len(Y)
    # Note: I modified plot_k_means from the original
    # lecture to return means and responsibilities
    # print "performing k-means..."
    # t0 = datetime.now()
    M, R = plot_k_means(X, len(set(Y)))
    # print "k-means elapsed time:", (datetime.now() - t0)
    # Exercise: Try different values of K and compare the evaluation metrics
    print "Purity:", purity(Y, R)
    print "DBI:", DBI(X, M, R)

    # plot the mean images
    # they should look like digits
    for k in xrange(len(M)):
=======
    print("Number of data points:", len(Y))
    M, R = plot_k_means(X, len(set(Y)))
    # Exercise: Try different values of K and compare the evaluation metrics
    print("Purity:", purity(Y, R))
    print("Purity 2 (hard clusters):", purity2(Y, R))
    print("DBI:", DBI(X, M, R))
    print("DBI 2 (hard clusters):", DBI2(X, R))

    # plot the mean images
    # they should look like digits
    for k in range(len(M)):
>>>>>>> upstream/master
        im = M[k].reshape(28, 28)
        plt.imshow(im, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main()
