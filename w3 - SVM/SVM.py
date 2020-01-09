import pandas as pd
from sklearn.svm import SVC

# Load data set from CSV
data = pd.read_csv('svm-data.csv', header=None)

X = data.loc[:, 1:2]
y = data.loc[:, 0]


# We are using linear model SVC
# Basic parameters of SVC class are: C factor and kernel type
clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)

result = sorted([x+1 for x in clf.support_])
print(result)