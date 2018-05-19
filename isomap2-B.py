import math, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')
plt.style.use('ggplot')

def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax


usecolors = []
for i in range(len(files)):
    usecolors.append('blue')


for i in range(len(files2)):
    usecolors.append('red')



def Plot2D(T, title, x, y, num_to_plot=40):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))

    # x_size = (max(T[:, x]) - min(T[:, x])) * 0.08
    # y_size = (max(T[:, y]) - min(T[:, y])) * 0.08
    x_size = (max(T[x]) - min(T[x])) * 0.08
    y_size = (max(T[y]) - min(T[y])) * 0.08
    # It also plots the full scatter:
    ax.scatter(T[:, x], T[:, y], c = usecolors, marker='.', alpha=0.7)


def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)

    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled


import sys
import glob
import errno
import imageio
from PIL import Image


fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/ALOI/32/32*.png'

fname2 = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/ALOI/32i/32*.png'

# examples good code: http://effbot.org/imagingbook/image.htm


files = glob.glob(fname)
files2 = glob.glob(fname2)

imglist = []
for name in files:
    # print(name)
    ineimage = imageio.imread(name)
    imglist.append(ineimage.reshape(-1))

for name in files2:
    # print(name)
    ineimage = imageio.imread(name)
    imglist.append(ineimage.reshape(-1))









# with next line we get df with one row for each image, and all image pixels
# flatten out in one row:
dfx = pd.DataFrame(imglist)

# scaleFeatures = True
# if scaleFeatures: df3dmf = scaleFeaturesDF(df)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=6, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neighbours = 6", 0, 1, num_to_plot=40)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=5, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neightbours=5", 0, 1, num_to_plot=40)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=4, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neightbours=4", 0, 1, num_to_plot=40)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=3, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neightbours=3", 0, 1, num_to_plot=40)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=2, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neightbours=2", 0, 1, num_to_plot=40)

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=1, n_components=3)
finalx = iso.fit_transform(dfx)
Plot2D(finalx, "MANIFOLDS - components 0-2 - neightbours=1", 0, 1, num_to_plot=40)


