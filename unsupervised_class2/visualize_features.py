# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams
from util import relu, error_rate, getKaggleMNIST, init_weights
from unsupervised import DBN
from rbm import RBM


def main(loadfile=None, savefile=None):
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    if loadfile:
        dbn = DBN.load(loadfile)
    else:
        dbn = DBN([1000, 750, 500, 10]) # AutoEncoder is default
<<<<<<< HEAD
        dbn = DBN([1000, 750, 500, 10], UnsupervisedModel=RBM)
        dbn.fit(Xtrain, pretrain_epochs=15)
=======
        # dbn = DBN([1000, 750, 500, 10], UnsupervisedModel=RBM)
        dbn.fit(Xtrain, pretrain_epochs=2)
>>>>>>> upstream/master

    if savefile:
        dbn.save(savefile)

    # first layer features
    # initial weight is D x M
<<<<<<< HEAD
    # W = dbn.hidden_layers[0].W.eval()
    # for i in xrange(dbn.hidden_layers[0].M):
    #     imgplot = plt.imshow(W[:,i].reshape(28, 28), cmap='gray')
    #     plt.show()
    #     should_quit = raw_input("Show more? Enter 'n' to quit\n")
    #     if should_quit == 'n':
    #         break

    # features learned in the last layer
    for k in xrange(dbn.hidden_layers[-1].M):
=======
    W = dbn.hidden_layers[0].W.eval()
    for i in range(dbn.hidden_layers[0].M):
        imgplot = plt.imshow(W[:,i].reshape(28, 28), cmap='gray')
        plt.show()
        should_quit = input("Show more? Enter 'n' to quit\n")
        if should_quit == 'n':
            break

    # features learned in the last layer
    for k in range(dbn.hidden_layers[-1].M):
>>>>>>> upstream/master
        # activate the kth node
        X = dbn.fit_to_input(k)
        imgplot = plt.imshow(X.reshape(28, 28), cmap='gray')
        plt.show()
        if k < dbn.hidden_layers[-1].M - 1:
<<<<<<< HEAD
            should_quit = raw_input("Show more? Enter 'n' to quit\n")
=======
            should_quit = input("Show more? Enter 'n' to quit\n")
>>>>>>> upstream/master
            if should_quit == 'n':
                break


if __name__ == '__main__':
    # to load a saved file
<<<<<<< HEAD
    main(loadfile='rbm15.npz')

    # to neither load nor save
    # main()
=======
    # main(loadfile='rbm15.npz')

    # to neither load nor save
    main()
>>>>>>> upstream/master

    # to save a trained unsupervised deep network
    # main(savefile='rbm15.npz')