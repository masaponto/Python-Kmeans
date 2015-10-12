#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file

import random


class KMeans:

    def __init__(self, k, n=10, d=0):
        self.k = k
        self.n = n
        self.d = d

    def find_nearest(self, x, centers):
        dists = [np.linalg.norm(x - v) for v in centers]
        return dists.index(min(dists))

    def fit(self, X):

        self.labels = [random.randint(0, self.k - 1) for x in X]
        b_labels = self.labels

        self.centers = [np.array([0.0 for i in range(len(X[0]))])
                        for n in range(self.k)]

        for i in range(self.n):
            for x, label in zip(X, self.labels):
                self.centers[label] += x

            for i, center in enumerate(self.centers):
                self.centers[i] = center / self.labels.count(i)

            for i, x in enumerate(X):
                self.labels[i] = self.find_nearest(x, self.centers)

            if self.labels == b_labels:
                break

            b_labels = self.labels

    def plot(self, X):
        import matplotlib.pyplot as plt

        colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']

        for i, center in enumerate(self.centers):
            plt.plot(center[0], center[1], colors[i] + 'o')

        for i, x in zip(self.labels, X):
            plt.plot(x[0], x[1], colors[i] + '.')

        plt.show()
        plt.savefig('figures/kmeans.png')


def main():

    db_name = 'iris'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    kmeans = KMeans(5, n=1000)
    kmeans.fit(X_train)
    kmeans.plot(X_train)


if __name__ == "__main__":
    main()
