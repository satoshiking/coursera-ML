from skimage.io import imread
from skimage import img_as_float
import numpy as np
import pandas as pd
import pylab
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio


# Load image as ndarray with 3 dimensions: x,y,rgb
image = imread('parrots.jpg')
print('Начальная картинка: число измерений =', image.ndim, 'форма = ', image.shape)

# Decreasing dimension of ndarray to 2 to make it compatible with classifier
image_reshaped  = np.reshape(image, (337962, 3))
print('Reshaped image: число измерений =', image_reshaped.ndim, ' форма = ', image_reshaped.shape)


# Define features as x*y, rgb ndarray
X = img_as_float(image_reshaped)

for n in range(8,21):
    print("N_clusters = %i" % n)
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=241).fit(X)

    df_X = pd.DataFrame(X, columns=['R', 'G', 'B'])
    df_X.insert(3, 'cluster',  kmeans.labels_, True)
    clusters = df_X['cluster'].unique()
    df_X_mean = df_X.copy()
    df_X_median = df_X.copy()

    # Coloring each pixel as mean or median pixel in each cluster
    for cluster in clusters:
        mean = df_X.loc[df_X['cluster'] == cluster].mean()
        median = df_X.loc[df_X['cluster'] == cluster].median()

        df_X_mean.loc[df_X_mean['cluster'] == cluster, 'R'] = mean[0]
        df_X_mean.loc[df_X_mean['cluster'] == cluster, 'G'] = mean[1]
        df_X_mean.loc[df_X_mean['cluster'] == cluster, 'B'] = mean[2]

        df_X_median.loc[df_X_median['cluster'] == cluster, 'R'] = median[0]
        df_X_median.loc[df_X_median['cluster'] == cluster, 'G'] = median[1]
        df_X_median.loc[df_X_median['cluster'] == cluster, 'B'] = median[2]

    # Count peak_signal_noise_ratio metric having MEAN color in clusters
    df_clustered_mean = df_X_mean.loc[:, ['R', 'G', 'B']]
    image_clustered_mean = df_clustered_mean.to_numpy()
    psnr_mean = peak_signal_noise_ratio(X, image_clustered_mean)
    print('  psnr(mean) = {:.2f}'.format(psnr_mean))

    # Count peak_signal_noise_ratio metric having MEDIAN color in clusters
    df_clustered_median = df_X_median.loc[:, ['R', 'G', 'B']]
    image_clustered_median = df_clustered_median.to_numpy()
    psnr_median = peak_signal_noise_ratio(X, image_clustered_median)
    print('  psnr(median) = {:.2f}'.format(psnr_median))

    if psnr_median >= 20 or psnr_mean >= 20:
        break


matplotlib.use('tkagg')
# plt.subplot(111)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("first")
# plt.show()


# pylab.imshow(image_float)
# io.show()
