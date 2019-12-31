import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# Load data from CSV and drop rows with 'na' values
df = pd.read_csv('titanic.csv', usecols=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
#print(df.isnull().sum())
#print(df.isna().sum())
print(df.shape)
df.dropna(inplace=True)
print(df.shape)

# Задаем обучающую выборку и ответы
X = df[['Pclass', 'Fare', 'Age', 'Sex']]
y = df[['Survived']]

# Coding text values into int format
my_dict = {}
label = LabelEncoder()
label.fit(X.Sex)  # define list of values for coding
my_dict['Sex'] = list(label.classes_)
print(my_dict)
X.loc[:, 'Sex'] = label.transform(X.loc[:, 'Sex'])  # replace text values with their codes
print(X.head())

# Choose DecisionTreeClassifier model and train it
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

# Try to make prediction with created model
predict = [[2, 20, 30, 1]]
print('predict = ', clf.predict(predict))

importances = clf.feature_importances_
print("The most important features:", importances)
