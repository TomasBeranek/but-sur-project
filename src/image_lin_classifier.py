import numpy as np
import matplotlib.pyplot as plt
from ikrlib import train_generative_linear_classifier

def PCA(train_t, train_n):
    # reshapes each image into vector
    train_t = np.r_[train_t].reshape(-1, 80*80)
    train_n = np.r_[train_n].reshape(-1, 80*80)

    # concat all images into vector of images
    X = np.r_[train_t, train_n]

    # groundtruth
    GT = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]

    # center data
    mean_face = np.mean(X, axis=0)
    X = X - mean_face

    # calculate PCA --  len(X) directions with highest variability
    V, S, U = np.linalg.svd(X, full_matrices=False)

    # transform data to len(X) (140) PCA bases
    X = X.dot(U.T)

    return (X, GT, U, mean_face)

class ModelImageLinClassifier():
    def __init__(self, dimensions):
        # how many leading PCA bases we are going to classify
        self.dimensions = dimensions


    def train(self, train_t, train_n):
        X, GT, self.U, self.mean_face = PCA(train_t, train_n)
        self.w, self.w0, _ = train_generative_linear_classifier(X[:,:self.dimensions], GT)


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
