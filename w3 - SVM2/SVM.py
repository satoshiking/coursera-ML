import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold


newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
)

X_raw = newsgroups.data
y = newsgroups.target

# set weight for every word in texts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_raw)

# list of all different words
feature_mapping = vectorizer.get_feature_names()

# Searching parameter C in range (10^-5, 10^-4, ... 10^5) with best accuracy on cross validation
# Takes long time to evaluate
"""
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)


print(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'])
print("The best accuracy %s is achieved with next params: %s" % (gs.best_score_, gs.best_params_))
"""

clf = SVC(C=1.0, kernel='linear', random_state=241)
clf.fit(X, y)

# Searching for 10 features(words) with higherst weight
data = clf.coef_.toarray()[0]
weight = [abs(w) for w in data]
word_indexes = [k for k, v in sorted(list(enumerate(weight)), key=lambda x: x[1], reverse=True)][:10]
words = sorted([feature_mapping[i] for i in word_indexes])

for word in words:
    print(word, end=' ')
