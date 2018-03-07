# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Continuous-observation HMM with no scaling, but allowing multiple observations
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
    def fit(self, X, max_iter=30, eps=10e-1):
=======
    def fit(self, X, max_iter=30, eps=1e0):
>>>>>>> upstream/master
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1] # assume each x is organized (T, D)

        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.R = np.ones((self.M, self.K)) / self.K # mixture proportions
<<<<<<< HEAD
        print "initial A:", self.A
        print "initial R:", self.R
        self.mu = np.zeros((self.M, self.K, D))
        for i in xrange(self.M):
            for k in xrange(self.K):
=======
        print("initial A:", self.A)
        print("initial R:", self.R)
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
>>>>>>> upstream/master
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i,k] = x[random_time_idx]
        self.sigma = np.zeros((self.M, self.K, D, D))
<<<<<<< HEAD
        for j in xrange(self.M):
            for k in xrange(self.K):
                self.sigma[j,k] = np.eye(D)

        costs = []
        for it in xrange(max_iter):
            if it % 1 == 0:
                print "it:", it
=======
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j,k] = np.eye(D)

        costs = []
        for it in range(max_iter):
            if it % 1 == 0:
                print("it:", it)
>>>>>>> upstream/master
            alphas = []
            betas = []
            gammas = []
            Bs = []
            # components = []
            P = np.zeros(N)

<<<<<<< HEAD
            for n in xrange(N):
=======
            for n in range(N):
>>>>>>> upstream/master
                x = X[n]
                T = len(x)

                # calculate B so we can lookup when updating alpha and beta
                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T)) # we'll need these later
<<<<<<< HEAD
                for j in xrange(self.M):
                    for t in xrange(T):
                        for k in xrange(self.K):
                            # print "sigma:", self.sigma[j,k]
=======
                for j in range(self.M):
                    for t in range(T):
                        for k in range(self.K):
>>>>>>> upstream/master
                            p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                            component[j,k,t] = p
                            B[j,t] += p
                Bs.append(B)

                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi*B[:,0]
<<<<<<< HEAD
                for t in xrange(1, T):
=======
                for t in range(1, T):
>>>>>>> upstream/master
                    alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
                P[n] = alpha[-1].sum()
                assert(P[n] <= 1)
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
<<<<<<< HEAD
                for t in xrange(T - 2, -1, -1):
=======
                for t in range(T - 2, -1, -1):
>>>>>>> upstream/master
                    beta[t] = self.A.dot(B[:,t+1] * beta[t+1])
                betas.append(beta)

                # update for Gaussians
                gamma = np.zeros((T, self.M, self.K))
<<<<<<< HEAD
                for t in xrange(T):
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    # print "alpabeta:", alphabeta
                    for j in xrange(self.M):
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        # mixture_j = component[j,:,t].sum()
                        for k in xrange(self.K):
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
                            # assert(np.abs(mixture_j - B[j,t]).sum() < 10e-10)
=======
                for t in range(T):
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    for j in range(self.M):
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        # mixture_j = component[j,:,t].sum()
                        for k in range(self.K):
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
>>>>>>> upstream/master
                gammas.append(gamma)

            cost = np.log(P).sum()
            costs.append(cost)

            # now re-estimate pi, A, R, mu, sigma
<<<<<<< HEAD
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in xrange(N)) / N
=======
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N
>>>>>>> upstream/master

            a_den = np.zeros((self.M, 1))
            a_num = 0
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
<<<<<<< HEAD
            for n in xrange(N):
=======
            for n in range(N):
>>>>>>> upstream/master
                x = X[n]
                T = len(x)
                B = Bs[n]
                gamma = gammas[n]

                # denominator for A
                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

                # numerator for A
                a_num_n = np.zeros((self.M, self.M))
<<<<<<< HEAD
                for i in xrange(self.M):
                    for j in xrange(self.M):
                        for t in xrange(T-1):
=======
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
>>>>>>> upstream/master
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
                a_num += a_num_n / P[n]


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
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros((self.M, self.K, D))
                sigma_num_n = np.zeros((self.M, self.K, D, D))
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
                            mu_num_n[j,k] += gamma[t,j,k] * x[t]

                            # update covariances
                            sigma_num_n[j,k] += gamma[t,j,k] * np.outer(x[t] - self.mu[j,k], x[t] - self.mu[j,k])
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            self.A = a_num / a_den
<<<<<<< HEAD
            # print self.A
            assert(np.all(self.A <= 1))

            # update R, mu, sigma
            for j in xrange(self.M):
                for k in xrange(self.K):
=======
            assert(np.all(self.A <= 1))

            # update R, mu, sigma
            for j in range(self.M):
                for k in range(self.K):
>>>>>>> upstream/master
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

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

    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))

        B = np.zeros((self.M, T))
<<<<<<< HEAD
        for j in xrange(self.M):
            for t in xrange(T):
                for k in xrange(self.K):
=======
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
>>>>>>> upstream/master
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        alpha[0] = self.pi*B[:,0]
<<<<<<< HEAD
        for t in xrange(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()

=======
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()

    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)

        # make the emission matrix B
        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        # perform Viterbi as usual
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi*B[:,0]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * B[j,t]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

>>>>>>> upstream/master
    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

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

    hmm = HMM(10)
    hmm.fit(signal.reshape(1, T))


def fake_signal(init=simple_init):
    signals = get_signals(N=10, T=10, init=init)
    # for signal in signals:
    #     for d in xrange(signal.shape[1]):
    #         plt.plot(signal[:,d])
    # plt.show()

    hmm = HMM(2, 2)
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
=======
    print("LL for actual params:", L)

    # print most likely state sequence
    print("Most likely state sequence for initial observation:")
    print(hmm.get_state_sequence(signals[0]))
>>>>>>> upstream/master

if __name__ == '__main__':
    # real_signal() # will break
    fake_signal(init=simple_init)
    # fake_signal(init=big_init) # will break

