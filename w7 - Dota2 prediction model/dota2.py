import pandas as pd
import numpy as np


features = pd.read_csv('./features.csv', index_col='match_id')
X = features.loc[:, 'start_time':'dire_first_ward_time']
print('X_shape=', X.shape, '\n')
print(X)



X_pick = np.zeros((features.shape[0], 112))
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
print('X_pick=', X_pick, '\n')



for i in range(112):
    col_name = 'hero_%i' % i
    X[col_name] = X_pick[:, i]

print(X.shape)
print(X)



