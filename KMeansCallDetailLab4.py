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
                    'DAT210x-master/Module5/Datasets/CDR.csv'

df = pd.read_csv(fname)
df.dropna(axis=0, how='any', inplace=True)
print(df.dtypes)
df['CallTime'] = pd.to_datetime(df['CallTime'])
InList = df['In'].unique()
user1 = df.loc[ df.In == InList[0] ]
# user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
# plt.show()

user1noweekend = user1.loc[(user1.DOW != 'Sat') & (user1.DOW != 'Sun') & (user1.CallTime < "17:00:00")]
user1noweekend.reset_index()
savdf = user1noweekend

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user1noweekend.TowerLon,user1noweekend.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekdays Calls (<5PM)')
plt.show()

keeph = user1noweekend[['In', 'Out' ]]
user1noweekend = user1noweekend[['TowerLon', 'TowerLat' ]]
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, init='random', n_init=100)
labels = model.fit_predict(user1noweekend)

# Now we can print and plot the centroids:
centroids = model.cluster_centers_
print(centroids)
ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='blue', alpha=0.5, linewidths=3, s=169)
plt.show()

# unique labels - 4 since there are 4 centroids
pd.unique(labels)
# sum the labels by centroid/cluster:
import collections
print(collections.Counter(labels))
print(collections.Counter(keeph['In']))
print(collections.Counter(keeph['Out']))

# cocrel road: 32.6972746,-96.9621843
# that is closest to the centroid # 2 - so that is the second most frequent one
# so find second most frequent phone number:

# round up time to hour - up or down whichever is closest
savdf['callh'] = savdf['CallTime'].apply(lambda x: x.hour)
savdf['callmin'] = savdf['CallTime'].apply(lambda x: x.minute)
savdf.loc[ savdf['callmin'] > 30, 'callh']  +=1
savdf.loc[labels==0, 'callh'].mean()    # 8 hour!






