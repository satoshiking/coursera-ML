import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


df = pd.read_csv('salary-train.csv')
print(df.info())
print(df.head())

# Make all texts lowercase
df['FullDescription']    = df['FullDescription'].str.lower()
df['LocationNormalized'] = df['LocationNormalized'].str.lower()
df['ContractTime']       = df['ContractTime'].str.lower()

# Replace all non digit and non letter symbols to ' '
df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df['FullDescription'] = df['FullDescription'].replace('[^*a-zA-Z0-9]', ' ', regex = True)
df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Replace empty values with 'nan'
df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

# Getting TF-IDF features from text
vectorizer = TfidfVectorizer(min_df=5)
X_descr_train = vectorizer.fit_transform(df['FullDescription'])

# One-hot coding of LocationNormalized and ContractTime columns
enc = DictVectorizer()
X_categ_train = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Put all features in one CRS Matrix
X_train = hstack([X_descr_train, X_categ_train])
y_train = df['SalaryNormalized']

# Train our Ridge classifier
clf = Ridge(alpha=1, random_state=241)
clf.fit(X_train, y_train)


# Prepare test data
df_test = pd.read_csv('salary-test-mini.csv')
df_test['FullDescription']    = df_test['FullDescription'].str.lower()
df_test['LocationNormalized'] = df_test['LocationNormalized'].str.lower()
df_test['ContractTime']       = df_test['ContractTime'].str.lower()

# Replace all non digit and non letter symbols to ' '
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df_test['FullDescription'] = df_test['FullDescription'].replace('[^*a-zA-Z0-9]', ' ', regex = True)
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Replace empty values with 'nan'
df_test['LocationNormalized'].fillna('nan', inplace=True)
df_test['ContractTime'].fillna('nan', inplace=True)

X_descr_test = vectorizer.transform(df_test['FullDescription'])
X_categ_test = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_descr_test, X_categ_test])

y_predicted = clf.predict(X_test)
for y in y_predicted:
    print("{:.2f}".format(y))