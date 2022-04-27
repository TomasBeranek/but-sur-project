from ikrlib import train_gmm, logpdf_gmm
import numpy as np
from numpy.random import randint
from math import prod
import os
import json

class ModelAudioGMM():
    def __init__(self, M_t=None, M_n=None, train_cycles=None, verbose=True):
        if verbose:
            print("NOTE: ModelAudioGMM initialized.")
        self.verbose = verbose
        self.M_t = M_t
        self.M_n = M_n
        self.train_cycles = train_cycles


    def train(self, train_t, train_n):
        # audio data preprocessing
        # cut off first 1.5s
        # TODO: iteration is very slow
        train_t = list(train_t)
        for i in range(0,len(train_t)):
           train_t[i] = train_t[i][150:]

        train_n = list(train_n)
        for i in range(0,len(train_n)):
           train_n[i] = train_n[i][150:]

        train_t = np.vstack(train_t)
        train_n = np.vstack(train_n)

        self.MUs_t = train_t[randint(1, len(train_t), self.M_t)]
        self.COVs_t = [np.cov(train_t.T)] * self.M_t
        self.Ws_t = np.ones(self.M_t) / self.M_t

        self.MUs_n = train_n[randint(1, len(train_n), self.M_n)]
        self.COVs_n = [np.cov(train_n.T)] * self.M_n
        self.Ws_n = np.ones(self.M_n) / self.M_n

        self.prev_TTL_t = None
        self.prev_TTL_n = None
        TTL_t = None
        TTL_n = None
        finished_t = False
        finished_n = False

        # run EM algorithm
        for i in range(self.train_cycles):
            self.Ws_t, self.MUs_t, self.COVs_t, TTL_t = train_gmm(train_t, self.Ws_t, self.MUs_t, self.COVs_t)
            self.Ws_n, self.MUs_n, self.COVs_n, TTL_n = train_gmm(train_n, self.Ws_n, self.MUs_n, self.COVs_n)

            if self.prev_TTL_t == None or abs(self.prev_TTL_t - TTL_t) > 10:
                self.prev_TTL_t = TTL_t
            else:
                finished_t = True

            if self.prev_TTL_n == None or abs(self.prev_TTL_n - TTL_n) > 10:
                self.prev_TTL_n = TTL_n
            else:
                finished_n = True

            if self.verbose:
                print("Iteration %d \tTotal log-likelyhood target: %f \t\t non-target: %f" % (i+1, TTL_t, TTL_n))

            if finished_t and finished_n:
                break

    def test(self, test, aprior_prob_t):
        P_t = aprior_prob_t
        P_n = 1 - P_t

        result = {}

        for file, sample in test.items():
            file = ''.join(file.split('/')[-1].split('.')[:-1])
            ll_t = logpdf_gmm(sample, self.Ws_t, self.MUs_t, self.COVs_t)
            ll_n = logpdf_gmm(sample, self.Ws_n, self.MUs_n, self.COVs_n)
            log_joint_prob_t = sum(ll_t) + np.log(P_t)
            log_joint_prob_n = sum(ll_n) + np.log(P_n)
            prediction_log_prob = log_joint_prob_t - log_joint_prob_n
            prediction = prediction_log_prob > 0
            result[file] = (prediction_log_prob, prediction)

        return result

    def save(self, path):
        # remove the file if it already exists
        if os.path.exists(path):
            os.remove(path)

        model_dict = {  "Ws_t":     self.Ws_t.tolist(),
                        "MUs_t":    self.MUs_t.tolist(),
                        "COVs_t":   self.COVs_t.tolist(),
                        "Ws_n":     self.Ws_n.tolist(),
                        "MUs_n":    self.MUs_n.tolist(),
                        "COVs_n":   self.COVs_n.tolist()}

        with open(path, "w") as file:
            json.dump(model_dict, file, indent=4)

    def load(self, path):
        with open(path) as file:
            model_dict = json.load(file)

        self.Ws_t   = np.array(model_dict['Ws_t'])
        self.MUs_t  = np.array(model_dict['MUs_t'])
        self.COVs_t = np.array(model_dict['COVs_t'])
        self.Ws_n   = np.array(model_dict['Ws_n'])
        self.MUs_n  = np.array(model_dict['MUs_n'])
        self.COVs_n = np.array(model_dict['COVs_n'])
