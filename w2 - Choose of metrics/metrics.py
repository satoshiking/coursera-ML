import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold


# Load data from standard sklearn library
data = datasets.load_boston()

# Standardize our dataset to have  average value = zero for each feature
data['data'] = scale(data['data'])

X = data['data']
y = data['target']
accuracy = []

for i in np.linspace(1, 10, num=200):
    # We are using KNeighborsRegressor class for regression
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    neigh.fit(X, y)

    # Count quality=squared_error of cross validation on splitted data
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(neigh, X, y, scoring='neg_mean_squared_error', cv=cv)

    print("p = %0.1f average squared error: %0.2f" % (i, scores.mean()))
    accuracy.append([i, scores.mean()])


accuracy.sort(key=lambda x: x[1], reverse=True)
print("The best quality in terms of squared error %0.2f is achieved with p=%0.1f." % (accuracy[0][1], accuracy[0][0]))
