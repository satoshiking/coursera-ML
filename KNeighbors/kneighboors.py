import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import scale


# Load data from CSV, there are no NULL or duplicated rows there
df = pd.read_csv('wine.data', header=None)
X = df.loc[:, 1:13]
y = df.loc[:, 0]

# Standardize our dataset to have zero average value for each feature
X = scale(X)

# Search for the best number of neighbors for best accuracy
accuracy = []
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X, y)

    # Define generator for splitting data on 5 blocks for learning and validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Count quality of cross validation on splitted data
    scores = cross_val_score(neigh, X, y, scoring='accuracy', cv=cv)
    # Find average accuracy between splitted blocks of data
    print("neighbors = %s  Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean(), scores.std() * 2))
    accuracy.append([i, scores.mean()])

accuracy.sort(key=lambda x: x[1], reverse=True)
print("The best accuracy %0.2f is achieved with %s neighbors." % (accuracy[0][1], accuracy[0][0]))
