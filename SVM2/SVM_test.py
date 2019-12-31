import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


newsgroups=datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(newsgroups.data)
y=newsgroups.target


clf2=SVC(kernel='linear',C=1.0, random_state=241)
clf2.fit(X,y)
coef=clf2.coef_
print(coef)


q=pd.DataFrame(coef.toarray()).transpose()
top10=abs(q).sort_values([0], ascending=False).head(10)
indices=[]
indices=top10.index
words=[]
print(indices)


for i in indices:
    feature_mapping=vectorizer.get_feature_names()
    words.append(feature_mapping[i])
print(sorted(words))

