import numpy as np
from numpy.core.arrayprint import printoptions
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pylab as pl


test_percent = 0.2

slices_per_window = 5
K = 3
TRESHOLD_NEIGHBOURS = 3


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


def main():
    # TODO: read csv into X and y4
    df = pd.read_csv('log3.csv')

    data_read = df.to_numpy()

   # preprocess
    for row in data_read:
        row[2] /= slices_per_window

    # TODO: split train dataset and test
    train, test = split(data_read, test_percent)

    # TODO: train KNN
    knn = KNN()

    X_train = np.array([row[: -1] for row in train])
    y_train = np.array([row[-1] for row in train])
    y_train = y_train.astype(int)

    X_test = np.array([row[: -1] for row in test])
    y_test = np.array([row[-1] for row in test])
    y_test = y_test.astype(int)

##############

    classColormap = ListedColormap(['#000000', '#FF0000'])
    pl.scatter([data_read[i][0] for i in range(len(data_read))],
               [data_read[i][1] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('0vs1.png')

    pl.scatter([data_read[i][0] for i in range(len(data_read))],
               [data_read[i][2] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('0vs2.png')

    pl.scatter([data_read[i][0] for i in range(len(data_read))],
               [data_read[i][3] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('0vs3.png')

    pl.scatter([data_read[i][1] for i in range(len(data_read))],
               [data_read[i][2] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('1vs2.png')

    pl.scatter([data_read[i][1] for i in range(len(data_read))],
               [data_read[i][3] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('1vs3.png')

    pl.scatter([data_read[i][2] for i in range(len(data_read))],
               [data_read[i][3] for i in range(len(data_read))],
               c=[data_read[i][-1] for i in range(len(data_read))],
               cmap=classColormap)
    pl.savefig('2vs3.png')

##########

    knn.fit(X_train, y_train)

    # TODO: obtain results

    res = knn.predict(X_test, K)

    final_result = np.zeros(len(y_test))

    for i in range(slices_per_window, len(y_test)):
        sum = 0
        for j in range(slices_per_window):
            sum += res[i - j]

        if sum < TRESHOLD_NEIGHBOURS:
            final_result[i] = 0
        else:
            final_result[i] = 1

    fails = 0
    success = 0
    for i in range(slices_per_window, len(y_test)):
        if (res[i] == y_test[i]):
            if res[i] == 1:
                print("[SUCCESS] caught virus!")
                success += 1
            else:
                print("[SUCCESS]")
                success += 1
        else:
            if res[i] == 0:
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
