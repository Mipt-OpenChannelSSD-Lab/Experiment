import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pylab as pl
import pandas as pd
import csv
import enum
import random


test_percent = 0.3


class KNN:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)

        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(((self.X_train - X[i]) ** 2).sum(axis=1))
        return dists

    def predict_labels(self, dists, k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred


def split(data, testPercent):
    trainData = []
    testData = []
    for row in data:
        if random.random() < testPercent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData


# random
def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        # Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random()*5.0, random.random()*5.0
        # Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(numberOfClassEl):
            data.append(
                [[random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)], classNum])
    return data


def showData(X, y):
    classColormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    pl.scatter([X[i][0] for i in range(len(X))],
               [X[i][1] for i in range(len(X))],
               c=[y],
               cmap=classColormap)
    pl.show()


def main1():
    data = generateData(40, 3)

    train, test = split(data, test_percent)

    X_train = np.array([row[0] for row in train])
    y_train = np.array([row[1] for row in train])

    X_test = np.array([row[0] for row in test])
    y_test = np.array([row[1] for row in test])

    print(X_train)
    print(y_train)

    showData(X_train, y_train)

    knn = KNN()

    knn.fit(X_train, y_train)

    # TODO: obtain results
    res = knn.predict(X_test, 3)
    for i in range(len(y_test)):
        if (res[i] == y_test[i]):
            print("[SUCCESS] element ", i, " passed!")
        else:
            print("[FAIL] element ", i, " failed!")

    return


def main():
    # TODO: read csv into X and y4
    df = pd.read_csv('data_log.csv')

    data_read = df.to_numpy()

    # TODO: split train dataset and test
    train, test = split(data_read, test_percent)

    # TODO: train KNN
    knn = KNN()

    X_train = np.array([row[:-1] for row in train])
    y_train = np.array([row[-1] for row in train])
    y_train = y_train.astype(int)

    X_test = np.array([row[:-1] for row in test])
    y_test = np.array([row[-1] for row in test])
    y_test = y_test.astype(int)

    knn.fit(X_train, y_train)

    # TODO: obtain results
    fails = 0
    success = 0
    for i in range(len(y_test)):
        res = knn.predict(X_test[i], 3)

        if (res[0] == y_test[i]):
            if res[0] == 1:
                print("[SUCCESS] caught virus!")
                success += 1
            else:
                print("[SUCCESS]")
        else:
            if res[0] == 0:
                print("[FAIL] uncaught virus!")
                fails += 1
            else:
                print("[FAIL] blocked SSD for nothing!")
                fails += 1

    print("[SUMMARY] fails:  ", fails, ", success: ", success)
    print("[SUMMARY] fails%: ", fails / (fails + success),
          "success%: ", success / (fails + success))

    return


if __name__ == "__main__":
    main()
