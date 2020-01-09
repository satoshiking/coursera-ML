import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load train and test data sets from CSV
train = pd.read_csv('perceptron-train.csv', header=None)
test = pd.read_csv('perceptron-test.csv', header=None)
X_train = train.loc[:, 1:2]
y_train = train.loc[:, 0]
X_test = test.loc[:, 1:2]
y_test = test.loc[:, 0]

# We are using linear model Perceptron with not normalized data
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# Count quality as accuracy on splitted data
predictions = clf.predict(X_test)
score = accuracy_score(y_test, predictions)
print("Accuracy of prediction without normalized data is {:.2%}".format(score))


# Normalize our data set knowing average and deviation for each feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Same model but with normalized data
clf_scaled = Perceptron()
clf_scaled.fit(X_train_scaled, y_train)
predictions_scaled = clf_scaled.predict(X_test_scaled)
score_scaled = accuracy_score(y_test, predictions_scaled)
print("Accuracy of prediction with having data normalized is {:.2%}".format(score_scaled))
print("Difference in accuracy is {:.3f}".format(score_scaled-score))
