import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef


# Reading data
df = pd.read_csv('close_prices.csv')
print(df.info())
print(df.head())

# Learning pca with 10 components
X = df.loc[:, 'AXP':'XOM']
pca = PCA(10)
pca.fit(X)

# How many components needed to cover 90% of dispersion ?
print("\nDispersion of each component: %s" % pca.explained_variance_ratio_)
i, disp = 0, 0
while disp <= 0.9:
    disp += pca.explained_variance_ratio_[i]
    i += 1
print("%s of components is enough to cover %s of dispersion\n" % (i, disp))

# Fit the model with X and apply the dimensionality reduction on X.
X_new = pca.transform(X)
#print("First component = %s" % pca.components_[0])
X_new_first_component = [x[0] for x in X_new]
print("X_new_first_component: %s \n" % X_new_first_component)

# Loading data of DJI
df_DJI = pd.read_csv('djia_index.csv')
y = df_DJI['^DJI']

# Finding Pearson product-moment correlation between 1st component and DJI
cor = corrcoef(X_new_first_component, y)
print("Pearson product-moment correlation between 1st component and DJI = {:.2f}".format(cor[0][1]))

# Finding company with max weight in first component
print("First component: %s" %pca.components_[0])
feature_with_max_weight = sorted(list(enumerate(pca.components_[0])), key=lambda x: x[1], reverse=True)[0][0]
print("Feature with max weight has %i number" % feature_with_max_weight)
features = list(df.columns)[1:]
print("All features: %s" % features)

print("This feature is %s" % features[feature_with_max_weight])
