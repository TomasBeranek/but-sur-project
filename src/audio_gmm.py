from ikrlib import train_gmm, logpdf_gmm
import numpy as np
from numpy.random import randint
from math import prod


class ModelAudioGMM():
    def __init__(self, M_t, M_n, train_cycles):
        print("NOTE: ModelAudioGMM initialized.")
        self.M_t = M_t
        self.M_n = M_n
        self.train_cycles = train_cycles


    def train(self, train_t, train_n):
        train_t = np.vstack(train_t)
        train_n = np.vstack(train_n)

        self.MUs_t = train_t[randint(1, len(train_t), self.M_t)]
        self.COVs_t = [np.cov(train_t.T)] * self.M_t
        self.Ws_t = np.ones(self.M_t) / self.M_t

        self.MUs_n = train_n[randint(1, len(train_n), self.M_n)]
        self.COVs_n = [np.cov(train_n.T)] * self.M_n
        self.Ws_n = np.ones(self.M_n) / self.M_n

        # run EM algorithm
        for i in range(self.train_cycles):
            self.Ws_t, self.MUs_t, self.COVs_t, TTL_t = train_gmm(train_t, self.Ws_t, self.MUs_t, self.COVs_t)
            self.Ws_n, self.MUs_n, self.COVs_n, TTL_n = train_gmm(train_n, self.Ws_n, self.MUs_n, self.COVs_n)
            print("Iteration %d \tTotal log-likelyhood target: %f \t\t non-target: %f" % (i+1, TTL_t, TTL_n))

    def test(self, test, aprior_prob_t):
        P_t = aprior_prob_t
        P_n = 1 - P_t

        prediction_log_prob = []

        for sample in test:
            ll_t = logpdf_gmm(sample, self.Ws_t, self.MUs_t, self.COVs_t)
            ll_n = logpdf_gmm(sample, self.Ws_n, self.MUs_n, self.COVs_n)
            log_joint_prob_t = sum(ll_t) + np.log(P_t)
            log_joint_prob_n = sum(ll_n) + np.log(P_n)
            prediction_log_prob.append(log_joint_prob_t - log_joint_prob_n)

        labels = np.array(prediction_log_prob) > 0
        return (prediction_log_prob, labels)
