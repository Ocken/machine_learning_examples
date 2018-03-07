<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master
# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
import pickle
import numpy as np
from util import get_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    X, Y = get_data()
<<<<<<< HEAD
    Ntrain = len(Y) / 4
=======
    Ntrain = len(Y) // 4
>>>>>>> upstream/master
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]

    model = RandomForestClassifier()
    model.fit(Xtrain, Ytrain)

    # just in case you're curious
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
<<<<<<< HEAD
    print "test accuracy:", model.score(Xtest, Ytest)
=======
    print("test accuracy:", model.score(Xtest, Ytest))
>>>>>>> upstream/master

    with open('mymodel.pkl', 'wb') as f:
        pickle.dump(model, f)
