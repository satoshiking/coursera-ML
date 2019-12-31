import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import sklearn

# Reading data
df = pd.read_csv('abalone.csv')
print(df.info())
print(df)

# Transform text column into set [-1, 0, 1]
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Define features and target value
X = df.loc[:, 'Sex':'ShellWeight']
y = df['Rings']


# Creating generator to split our data into 5 parts
cv = KFold(n_splits=5, shuffle=True, random_state=1)

for i in range(1,51):
    # Creating and learning our Random Forest classifier
    rf = RandomForestRegressor(n_estimators=i, random_state=1)
    rf.fit(X,y)

    y_pred = rf.predict(X)
    # score measure = coefficient of determination
    score_r2 = r2_score(y, y_pred)

    # Count quality of cross validation on splitted data with with coefficient of determination score measure
    scores = cross_val_score(rf, X, y, scoring='r2', cv=cv)

    # Find average accuracy between splitted blocks of data
    print("With {} forests we have score = {:.2f}".format(i, scores.mean()))

    if scores.mean() >= 0.52:
        break
