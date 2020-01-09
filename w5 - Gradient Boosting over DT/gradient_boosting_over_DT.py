import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


df = pd.read_csv('gbm-data.csv')
data = df.values

y = data[:, 0]
X = data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
for learning_rate in [0.2]:
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
    clf.fit(X_train, y_train)

    test_loss = np.empty(250)
    for m, y_decision in enumerate(clf.staged_decision_function(X_test)):
        y_pred_test = 1.0 / (1.0 + np.exp(-y_decision))
        test_loss[m] = log_loss(y_test, y_pred_test)
        if test_loss[m] == 0.5301645204906471:
            print('m={} min={:.2f}'.format(m, test_loss[m]))
    # print(test_loss.min())

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    # plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)

test_loss = log_loss(y_test, clf.predict_proba(X_test))
print("test_loss={:.2f}".format(test_loss))
