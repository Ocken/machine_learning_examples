# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# theano scan example: calculate fibonacci
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


N = T.iscalar('N')

def recurrence(n, fn_1, fn_2):
<<<<<<< HEAD
	return fn_1 + fn_2, fn_1

outputs, updates = theano.scan(
	fn=recurrence,
	sequences=T.arange(N),
	n_steps=N,
	outputs_info=[1., 1.]
)

fibonacci = theano.function(
	inputs=[N],
	outputs=outputs,
=======
  return fn_1 + fn_2, fn_1

outputs, updates = theano.scan(
  fn=recurrence,
  sequences=T.arange(N),
  n_steps=N,
  outputs_info=[1., 1.]
)

fibonacci = theano.function(
  inputs=[N],
  outputs=outputs,
>>>>>>> upstream/master
)

o_val = fibonacci(8)

<<<<<<< HEAD
print "output:", o_val
=======
print("output:", o_val)
>>>>>>> upstream/master
