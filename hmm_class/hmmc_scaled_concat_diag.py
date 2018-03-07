# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# https://lazyprogrammer.me
# Continuous-observation HMM with scaling and multiple observations (treated as concatenated sequence)
<<<<<<< HEAD
=======
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


>>>>>>> upstream/master
import wave
import numpy as np
import matplotlib.pyplot as plt

from generate_c import get_signals, big_init, simple_init
from scipy.stats import multivariate_normal as mvn

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:
    def __init__(self, M, K):
        self.M = M # number of hidden states
        self.K = K # number of Gaussians
    
<<<<<<< HEAD
    def fit(self, X, max_iter=25, eps=10e-2):
=======
    def fit(self, X, max_iter=25, eps=1e-1):
>>>>>>> upstream/master
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # concatenate sequences in X and determine start/end positions
        sequenceLengths = []
        for x in X:
            sequenceLengths.append(len(x))
        Xc = np.concatenate(X)
        T = len(Xc)
        startPositions = np.zeros(len(Xc), dtype=np.bool)
        endPositions = np.zeros(len(Xc), dtype=np.bool)
        startPositionValues = []
        last = 0
        for length in sequenceLengths:
            startPositionValues.append(last)
            startPositions[last] = 1
            if last > 0:
                endPositions[last - 1] = 1
            last += length

        D = X[0].shape[1] # assume each x is organized (T, D)

        # randomly initialize all parameters
        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
        self.mu = np.zeros((self.M, self.K, D))
<<<<<<< HEAD
        for i in xrange(self.M):
            for k in xrange(self.K):
=======
        for i in range(self.M):
            for k in range(self.K):
>>>>>>> upstream/master
                random_idx = np.random.choice(T)
                self.mu[i,k] = Xc[random_idx]
        self.sigma = np.ones((self.M, self.K, D))

        # main EM loop
        costs = []
<<<<<<< HEAD
        for it in xrange(max_iter):
            if it % 1 == 0:
                print "it:", it
=======
        for it in range(max_iter):
            if it % 1 == 0:
                print("it:", it)
>>>>>>> upstream/master
            
            scale = np.zeros(T)

            # calculate B so we can lookup when updating alpha and beta
            B = np.zeros((self.M, T))
            component = np.zeros((self.M, self.K, T)) # we'll need these later
<<<<<<< HEAD
            for j in xrange(self.M):
                for k in xrange(self.K):
=======
            for j in range(self.M):
                for k in range(self.K):
>>>>>>> upstream/master
                    p = self.R[j,k] * mvn.pdf(Xc, self.mu[j,k], self.sigma[j,k])
                    component[j,k,:] = p
                    B[j,:] += p


            alpha = np.zeros((T, self.M))
            alpha[0] = self.pi*B[:,0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
<<<<<<< HEAD
            for t in xrange(1, T):
=======
            for t in range(1, T):
>>>>>>> upstream/master
                if startPositions[t] == 0:
                    alpha_t_prime = alpha[t-1].dot(self.A) * B[:,t]
                else:
                    alpha_t_prime = self.pi * B[:,t]
                scale[t] = alpha_t_prime.sum()
                alpha[t] = alpha_t_prime / scale[t]
            logP = np.log(scale).sum()

            beta = np.zeros((T, self.M))
            beta[-1] = 1
<<<<<<< HEAD
            for t in xrange(T - 2, -1, -1):
=======
            for t in range(T - 2, -1, -1):
>>>>>>> upstream/master
                if startPositions[t + 1] == 1:
                    beta[t] = 1
                else:
                    beta[t] = self.A.dot(B[:,t+1] * beta[t+1]) / scale[t+1]

            # update for Gaussians
            gamma = np.zeros((T, self.M, self.K))
<<<<<<< HEAD
            for t in xrange(T):
                alphabeta = alpha[t,:].dot(beta[t,:])
                for j in xrange(self.M):
                    factor = alpha[t,j] * beta[t,j] / alphabeta
                    for k in xrange(self.K):
=======
            for t in range(T):
                alphabeta = alpha[t,:].dot(beta[t,:])
                for j in range(self.M):
                    factor = alpha[t,j] * beta[t,j] / alphabeta
                    for k in range(self.K):
>>>>>>> upstream/master
                        gamma[t,j,k] = factor * component[j,k,t] / B[j,t]

            costs.append(logP)

            # now re-estimate pi, A, R, mu, sigma
            self.pi = np.sum((alpha[t] * beta[t]) for t in startPositionValues) / len(startPositionValues)

            a_den = np.zeros((self.M, 1)) # prob don't need this
            a_num = np.zeros((self.M, self.M))
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D))



            nonEndPositions = (1 - endPositions).astype(np.bool)
            a_den += (alpha[nonEndPositions] * beta[nonEndPositions]).sum(axis=0, keepdims=True).T

            # numerator for A
<<<<<<< HEAD
            for i in xrange(self.M):
                for j in xrange(self.M):
                    for t in xrange(T-1):
=======
            for i in range(self.M):
                for j in range(self.M):
                    for t in range(T-1):
>>>>>>> upstream/master
                        if endPositions[t] != 1:
                            a_num[i,j] += alpha[t,i] * beta[t+1,j] * self.A[i,j] * B[j,t+1] / scale[t+1]
            self.A = a_num / a_den


            # update mixture components
            r_num_n = np.zeros((self.M, self.K))
            r_den_n = np.zeros(self.M)
<<<<<<< HEAD
            for j in xrange(self.M):
                for k in xrange(self.K):
                    for t in xrange(T):
=======
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
>>>>>>> upstream/master
                        r_num_n[j,k] += gamma[t,j,k]
                        r_den_n[j] += gamma[t,j,k]
            r_num = r_num_n
            r_den = r_den_n

            mu_num_n = np.zeros((self.M, self.K, D))
            sigma_num_n = np.zeros((self.M, self.K, D))
<<<<<<< HEAD
            for j in xrange(self.M):
                for k in xrange(self.K):
                    for t in xrange(T):
=======
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
>>>>>>> upstream/master
                        # update means
                        mu_num_n[j,k] += gamma[t,j,k] * Xc[t]

                        # update covariances
                        sigma_num_n[j,k] += gamma[t,j,k] * (Xc[t] - self.mu[j,k])**2
            mu_num = mu_num_n
            sigma_num = sigma_num_n


            # update R, mu, sigma
<<<<<<< HEAD
            for j in xrange(self.M):
                for k in xrange(self.K):
=======
            for j in range(self.M):
                for k in range(self.K):
>>>>>>> upstream/master
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k] + np.ones(D)*eps
            assert(np.all(self.R <= 1))
            assert(np.all(self.A <= 1))
<<<<<<< HEAD
        print "A:", self.A
        print "mu:", self.mu
        print "sigma:", self.sigma
        print "R:", self.R
        print "pi:", self.pi
=======
        print("A:", self.A)
        print("mu:", self.mu)
        print("sigma:", self.sigma)
        print("R:", self.R)
        print("pi:", self.pi)
>>>>>>> upstream/master

        plt.plot(costs)
        plt.show()

    def log_likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        scale = np.zeros(T)
        B = np.zeros((self.M, T))
<<<<<<< HEAD
        for j in xrange(self.M):
            for k in xrange(self.K):
=======
        for j in range(self.M):
            for k in range(self.K):
>>>>>>> upstream/master
                p = self.R[j,k] * mvn.pdf(x, self.mu[j,k], self.sigma[j,k])
                B[j,:] += p

        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi*B[:,0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
<<<<<<< HEAD
        for t in xrange(1, T):
=======
        for t in range(1, T):
>>>>>>> upstream/master
            alpha_t_prime = alpha[t-1].dot(self.A) * B[:,t]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

<<<<<<< HEAD
=======
    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)

        # make the emission matrix B
        logB = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = np.log(self.R[j,k]) + mvn.logpdf(x[t], self.mu[j,k], self.sigma[j,k])
                    logB[j,t] += p
        print("logB:", logB)

        # perform Viterbi as usual
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))

        # smooth pi in case it is 0
        pi = self.pi + 1e-10
        pi /= pi.sum()

        delta[0] = np.log(pi) + logB[:,0]
        for t in range(1, T):
            for j in range(self.M):
                next_delta = delta[t-1] + np.log(self.A[:,j])
                delta[t,j] = np.max(next_delta) + logB[j,t]
                psi[t,j] = np.argmax(next_delta)

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

>>>>>>> upstream/master
    def log_likelihood_multi(self, X):
        return np.array([self.log_likelihood(x) for x in X])

    def set(self, pi, A, R, mu, sigma):
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        M, K = R.shape
        self.M = M
        self.K = K


def real_signal():
    spf = wave.open('helloworld.wav', 'r')

    #Extract Raw Audio from Wav File
    # If you right-click on the file and go to "Get Info", you can see:
    # sampling rate = 16000 Hz
    # bits per sample = 16
    # The first is quantization in time
    # The second is quantization in amplitude
    # We also do this for images!
    # 2^16 = 65536 is how many different sound levels we have
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()
    hmm = HMM(5, 3)
    hmm.fit(signal.reshape(1, T, 1), max_iter=35)
<<<<<<< HEAD
    print "LL for fitted params:", hmm.log_likelihood(signal.reshape(T, 1))
=======
    print("LL for fitted params:", hmm.log_likelihood(signal.reshape(T, 1)))
>>>>>>> upstream/master


def fake_signal(init=big_init):
    signals = get_signals(init=init)
    # for signal in signals:
    #     for d in xrange(signal.shape[1]):
    #         plt.plot(signal[:,d])
    # plt.show()

    hmm = HMM(5, 3)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
<<<<<<< HEAD
    print "LL for fitted params:", L
=======
    print("LL for fitted params:", L)
>>>>>>> upstream/master

    # test in actual params
    _, _, _, pi, A, R, mu, sigma = init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
<<<<<<< HEAD
    print "LL for actual params:", L

if __name__ == '__main__':
    real_signal()
    # fake_signal()
=======
    print("LL for actual params:", L)

    # print most likely state sequence
    print("Most likely state sequence for initial observation:")
    print(hmm.get_state_sequence(signals[0]))

if __name__ == '__main__':
    # real_signal()
    fake_signal()
>>>>>>> upstream/master

