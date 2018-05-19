#
#
#   k-means
#
import math, random
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')
plt.style.use('ggplot')



# initialization methods, including rolling your own and in the
# positions as an NDArray shaped as [n_clusters, n_features].
#  job at hand is still taking too long, SciKit-Learn's MiniBatchKMeans
# further optimizes the process for you.
#
# K-Means is only really suitable when you have a good estimate of the number
# clusters that exist in your unlabeled data. There are many estimation
# techniques for approximating the correct number of clusters, but you'll have
# to get that number before running K-Means. Even if you do have the right
# number of clusters selected, the result produced by K-Means can vary depending
# on the initial centroid placement. So if you need the same results produced
# each time, your centroid seeding technique also needs to be able to reliably
# produce the same placement given the same data. Due to the centroid seed
# placement having so much of an effect on your clustering outcome, you have
# to be careful since it is possible to have centroids with only a single
# sample assigned to them, or even no samples assigned to them in the worst case scenario.

# Two other key characteristics of K-Means are that it assumes your samples
# are length normalized, and as such, is sensitive to feature scaling. It also
# assumes that the cluster sizes are roughly spherical and similar; this way,
# the nearest centroid is always the correct assignment.
pd.set_option('display.width', 300)
fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module5/Datasets/gambling2.csv'

df = pd.read_csv(fname, index_col=0)
df.dropna(axis=0, how='any', inplace=True)
print(df.dtypes)
vname = 'Date'
df[vname] = pd.to_datetime(df[vname])
df['Updated.On'] = pd.to_datetime(df['Updated.On'])
df.drop(['X'], 1, inplace=True)
df.Location = df.Location.apply(lambda x: re.sub("[()]", "", str(x)))
df['Longitude'] = df['Location'].apply(lambda x: x.split(",")[0]).astype(float)
df['Latitude'] = df['Location'].apply(lambda x: x.split(",")[1]).astype(float)

df3 = df[['Longitude', 'Latitude', 'Date']]


def doKMeans(df):
    # Let's plot your data with a '.' marker, a 0.3 alpha at the Longitude,
    # and Latitude locations in your dataset. Longitude = x, Latitude = y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

    # TODO: Filter `df` using indexing so it only contains Longitude and Latitude,
    # since the remaining columns aren't really applicable for this lab:
    #
    # .. your code here ..
    df = df[['Longitude', 'Latitude']]

    # TODO: Use K-Means to try and find seven cluster centers in this df.
    # Be sure to name your kmeans model `model` so that the printing works.
    #
    # .. your code here ..
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=7, init='random', n_init=100)
    labels = model.fit_predict(df)

    # Now we can print and plot the centroids:
    centroids = model.cluster_centers_
    print(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='blue', alpha=0.5, linewidths=3, s=169)
    plt.show()


doKMeans(df)
df11 = df3.loc[df3['Date'] > '2011-01-01']
doKMeans(df11)





