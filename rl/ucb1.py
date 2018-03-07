# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# https://books.google.ca/books?id=_ATpBwAAQBAJ&lpg=PA201&ots=rinZM8jQ6s&dq=hoeffding%20bound%20gives%20probability%20%22greater%20than%201%22&pg=PA201#v=onepage&q&f=false
<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

>>>>>>> upstream/master
import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import run_experiment as run_experiment_eps


class Bandit:
  def __init__(self, m):
    self.m = m
    self.mean = 0
    self.N = 0

  def pull(self):
    return np.random.randn() + self.m

  def update(self, x):
    self.N += 1
    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x


def ucb(mean, n, nj):
<<<<<<< HEAD
  return mean + np.sqrt(2*np.log(n) / (nj + 1e-2))
=======
  if nj == 0:
    return float('inf')
  return mean + np.sqrt(2*np.log(n) / nj)
>>>>>>> upstream/master


def run_experiment(m1, m2, m3, N):
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

  data = np.empty(N)
  
<<<<<<< HEAD
  for i in xrange(N):
    # optimistic initial values
=======
  for i in range(N):
>>>>>>> upstream/master
    j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)

    # for the plot
    data[i] = x
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

<<<<<<< HEAD
=======
  # for b in bandits:
  #   print("bandit nj:", b.N)

>>>>>>> upstream/master
  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

<<<<<<< HEAD
  for b in bandits:
    print b.mean
=======
  # for b in bandits:
  #   print(b.mean)
>>>>>>> upstream/master

  return cumulative_average

if __name__ == '__main__':
<<<<<<< HEAD
  c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
  oiv = run_experiment(1.0, 2.0, 3.0, 100000)

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='ucb1')
=======
  eps = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
  ucb = run_experiment(1.0, 2.0, 3.0, 100000)

  # log scale plot
  plt.plot(eps, label='eps = 0.1')
  plt.plot(ucb, label='ucb1')
>>>>>>> upstream/master
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
<<<<<<< HEAD
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='ucb1')
=======
  plt.plot(eps, label='eps = 0.1')
  plt.plot(ucb, label='ucb1')
>>>>>>> upstream/master
  plt.legend()
  plt.show()

