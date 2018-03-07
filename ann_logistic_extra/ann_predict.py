<<<<<<< HEAD
import numpy as np
from process import get_data

X, Y = get_data()
=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from process import get_data

X, Y, _, _ = get_data()
>>>>>>> upstream/master

# randomly initialize weights
M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
<<<<<<< HEAD
=======
print("P_Y_given_X.shape:", P_Y_given_X.shape)
>>>>>>> upstream/master
predictions = np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

<<<<<<< HEAD
print "Score:", classification_rate(Y, predictions)
=======
print("Score:", classification_rate(Y, predictions))
>>>>>>> upstream/master
