# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a Naive Bayes classifier on MNIST data.
<<<<<<< HEAD
=======
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master

import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):
<<<<<<< HEAD
    def fit(self, X, Y, smoothing=10e-3):
        # N, D = X.shape
=======
    def fit(self, X, Y, smoothing=1e-2):
>>>>>>> upstream/master
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
<<<<<<< HEAD
            # assert(self.gaussians[c]['mean'].shape[0] == D)
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
        # print "gaussians:", self.gaussians
        # print "priors:", self.priors
=======
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
>>>>>>> upstream/master

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
<<<<<<< HEAD
        for c, g in self.gaussians.iteritems():
            # print "c:", c
=======
        for c, g in iteritems(self.gaussians):
>>>>>>> upstream/master
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, Y = get_data(10000)
<<<<<<< HEAD
    Ntrain = len(Y) / 2
=======
    Ntrain = len(Y) // 2
>>>>>>> upstream/master
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = NaiveBayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
<<<<<<< HEAD
    print "Training time:", (datetime.now() - t0)

    t0 = datetime.now()
    print "Train accuracy:", model.score(Xtrain, Ytrain)
    print "Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain)

    t0 = datetime.now()
    print "Test accuracy:", model.score(Xtest, Ytest)
    print "Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest)
=======
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
>>>>>>> upstream/master
