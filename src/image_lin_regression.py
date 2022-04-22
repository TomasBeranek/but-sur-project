import numpy as np
import matplotlib.pyplot as plt
from ikrlib import train_linear_logistic_regression
from image_lin_classifier import PCA

class ModelImageLinRegression():
    def __init__(self, dimensions, verbose):
        # how many leading PCA bases we are going to classify
        self.dimensions = dimensions
        self.verbose = verbose


    def train(self, train_t, train_n, epochs, init_w, init_w0):
        self.w = init_w
        self.w0 = init_w0

        X, GT, self.U, self.mean_face = PCA(train_t, train_n)
        for i in range(epochs):
            self.w, self.w0 = train_linear_logistic_regression(X[:,:self.dimensions], GT, self.w, self.w0)
            if self.verbose:
                print("Epoch %d      w: %f     w0: %f" % (i, self.w[0], self.w0))

    def test(self, test):
        result = {}

        for file, image in test.items():
            # reshapes each image into vector
            image = np.r_[image].reshape(-1, 80*80)

            # transform test data
            image = (image - self.mean_face).dot(self.U.T)

            # make predictions
            score = image[:,:self.dimensions].dot(self.w) + self.w0
            result[file] = (score, score > 0)

        return result
