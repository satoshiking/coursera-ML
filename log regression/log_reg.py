import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Load data set from CSV
data = pd.read_csv('data-logistic.csv', header=None)
X_ = data.values[:, 1:]
y_ = data.values[:, :1].T[0]

def distance(a, b):
    return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))

def log_regression(X, y, k, w, C, epsilon, max_iter):
    w1, w2 = w
    for i in range(max_iter):
        w1new = w1 + k*np.mean(y * X[:, 0] * (1 - (1 / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k*C*w1
        w2new = w2 + k*np.mean(y * X[:, 1] * (1 - (1 / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k*C*w2
        if distance((w1new, w2new), (w1, w2)) < epsilon:
            break
        w1, w2 = w1new, w2new

    predictions = []
    for i in range(len(X)):
        t1 = -w1*X[i, 0] - w2*X[i, 1]
        s = 1.0 / (1 + math.exp(t1))
        predictions.append(s)

    return predictions

p0 = log_regression(X_, y_, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
p1 = log_regression(X_, y_, 0.1, [0.0, 0.0], 10, 0.00001, 10000)

print("AUC-ROC with no regularization = {:.3f}".format(roc_auc_score(y_, p0)))
print("AUC-ROC with L2 regularization = {:.3f}".format(roc_auc_score(y_, p1)))
