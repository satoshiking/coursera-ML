import skimage.io as io
from skimage.io import imread
from skimage import img_as_float
import numpy as np
import pandas as pd
import pylab
import matplotlib
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans


matplotlib.use( 'tkagg' )

image = imread('parrots.jpg')
image_float = img_as_float(image)
print(len(image_float))
print(len(image_float[0]))
print(len(image_float[0][0]))

print(image_float[0])
print()
print(image_float[0][0])


# Define features as x,r coordinates and RGB for each pixel
XYR = image_float[0][:,0:3]
G = image_float[1][:,2]
B = image_float[2][:,2]
XYRGB = np.column_stack((XYR, G, B ))
X = XYRGB
# print(X)

y_pred = KMeans(init='k-means++', random_state=241).fit_predict(X)
# print(len(y_pred))
# print(y_pred)
plt.subplot(111)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("first")
# plt.show()


# pylab.imshow(image_float)
# io.show()