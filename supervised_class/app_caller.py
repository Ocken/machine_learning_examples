<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master
# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
import requests
import numpy as np
import matplotlib.pyplot as plt
from util import get_data

# make a prediction from our own server!
# in reality this could be coming from any client

X, Y = get_data()
N = len(Y)
while True:
    i = np.random.choice(N)
    r = requests.post("http://localhost:8888/predict", data={'input': X[i]})
<<<<<<< HEAD
    j = r.json()
    print j
    print "target:", Y[i]

    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.show()

    response = raw_input("Continue? (Y/n)\n")
=======
    print("RESPONSE:")
    print(r.content)
    j = r.json()
    print(j)
    print("target:", Y[i])

    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title("Target: %d, Prediction: %d" % (Y[i], j['prediction']))
    plt.show()

    response = input("Continue? (Y/n)\n")
>>>>>>> upstream/master
    if response in ('n', 'N'):
        break
