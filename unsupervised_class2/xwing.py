# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weights
<<<<<<< HEAD
=======
from autoencoder import momentum_updates
>>>>>>> upstream/master

# new additions used to compare purity measure using GMM
import os
import sys
sys.path.append(os.path.abspath('..'))
from unsupervised_class.kmeans_mnist import purity
<<<<<<< HEAD
# from unsupervised_class.gmm import gmm
=======
>>>>>>> upstream/master
from sklearn.mixture import GaussianMixture

class Layer(object):
    def __init__(self, m1, m2):
        W = init_weights((m1, m2))
<<<<<<< HEAD
        bi = np.zeros(m2)
        bo = np.zeros(m1)
=======
        bi = np.zeros(m2, dtype=np.float32)
        bo = np.zeros(m1, dtype=np.float32)
>>>>>>> upstream/master
        self.W = theano.shared(W)
        self.bi = theano.shared(bi)
        self.bo = theano.shared(bo)
        self.params = [self.W, self.bi, self.bo]

    def forward(self, X):
        return T.nnet.sigmoid(X.dot(self.W) + self.bi)

    def forwardT(self, X):
        return T.nnet.sigmoid(X.dot(self.W.T) + self.bo)


class DeepAutoEncoder(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=50, batch_sz=100, show_fig=False):
<<<<<<< HEAD
        N, D = X.shape
        n_batches = N / batch_sz
=======
        # cast hyperparams
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)

        N, D = X.shape
        n_batches = N // batch_sz
>>>>>>> upstream/master

        mi = D
        self.layers = []
        self.params = []
        for mo in self.hidden_layer_sizes:
            layer = Layer(mi, mo)
            self.layers.append(layer)
            self.params += layer.params
            mi = mo

        X_in = T.matrix('X')
        X_hat = self.forward(X_in)

        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).mean()
        cost_op = theano.function(
            inputs=[X_in],
            outputs=cost,
        )

<<<<<<< HEAD
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        grads = T.grad(cost, self.params)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]
=======
        updates = momentum_updates(cost, self.params, mu, learning_rate)
>>>>>>> upstream/master
        train_op = theano.function(
            inputs=[X_in],
            outputs=cost,
            updates=updates,
        )

        costs = []
<<<<<<< HEAD
        for i in xrange(epochs):
            print "epoch:", i
            X = shuffle(X)
            for j in xrange(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                c = train_op(batch)
                if j % 100 == 0:
                    print "j / n_batches:", j, "/", n_batches, "cost:", c
=======
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                c = train_op(batch)
                if j % 100 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
>>>>>>> upstream/master
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)

        self.map2center = theano.function(
            inputs=[X],
            outputs=Z,
        )

<<<<<<< HEAD
        for i in xrange(len(self.layers)-1, -1, -1):
=======
        for i in range(len(self.layers)-1, -1, -1):
>>>>>>> upstream/master
            Z = self.layers[i].forwardT(Z)

        return Z


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dae = DeepAutoEncoder([500, 300, 2])
    dae.fit(Xtrain)
    mapping = dae.map2center(Xtrain)
    plt.scatter(mapping[:,0], mapping[:,1], c=Ytrain, s=100, alpha=0.5)
    plt.show()

    # purity measure from unsupervised machine learning pt 1
<<<<<<< HEAD
    gmm = GaussianMixture(n_components=10)
    gmm.fit(Xtrain)
    responsibilities_full = gmm.predict_proba(Xtrain)
    print "full purity:", purity(Ytrain, responsibilities_full)

    gmm.fit(mapping)
    responsibilities_reduced = gmm.predict_proba(mapping)
    print "reduced purity:", purity(Ytrain, responsibilities_reduced)
=======
    # NOTE: this will take a long time (i.e. just leave it overnight)
    gmm = GaussianMixture(n_components=10)
    gmm.fit(Xtrain)
    print("Finished GMM training")
    responsibilities_full = gmm.predict_proba(Xtrain)
    print("full purity:", purity(Ytrain, responsibilities_full))

    gmm.fit(mapping)
    responsibilities_reduced = gmm.predict_proba(mapping)
    print("reduced purity:", purity(Ytrain, responsibilities_reduced))
>>>>>>> upstream/master


if __name__ == '__main__':
    main()

