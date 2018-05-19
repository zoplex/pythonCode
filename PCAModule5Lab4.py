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
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

from sklearn import preprocessing
from sklearn.decomposition import PCA

# You might need to import more modules here..
# .. your code here ..

matplotlib.style.use('ggplot')  # Look Pretty
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')
plt.style.use('ggplot')
pd.set_option('display.width', 300)


PLOT_TYPE_TEXT = False    # If you'd like to see indices
PLOT_VECTORS = True       # If you'd like to see your original features in P.C.-Space


def drawVectors(transformed_features, components_, columns, plt):
    num_columns = len(columns)

    # This function will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:, 0])
    yvector = components_[1] * max(transformed_features[:, 1])

    ## Visualize projections

    # Sort each column by its length. These are your *original*
    # columns, not the principal components.
    important_features = {columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2) for i in range(num_columns)}
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Projected Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75, zorder=600000)
        plt.text(xvector[i] * 1.2, yvector[i] * 1.2, list(columns)[i], color='b', alpha=0.75, zorder=600000)

    return ax


def doPCA(data, dimensions=2):
    model = PCA(n_components=dimensions, svd_solver='randomized', random_state=7)
    model.fit(data)
    return model


def doKMeans(data, num_clusters=0):
    # TODO: Do the KMeans clustering here, passing in the # of clusters parameter
    # and fit it against your data. Then, return a tuple containing the cluster
    # centers and the labels.
    #
    # Hint: Just like with doPCA above, you will have to create a variable called
    # `model`, which will be a SKLearn K-Means model for this to work.
    model = KMeans( n_clusters=num_clusters)
    labels = model.fit_predict(data)
    return model.cluster_centers_, model.labels_


# Attribute Information:
#
# 1) FRESH: annual spending (m.u.) on fresh products (Continuous);
# 2) MILK: annual spending (m.u.) on milk products (Continuous);
# 3) GROCERY: annual spending (m.u.)on grocery products (Continuous);
# 4) FROZEN: annual spending (m.u.)on frozen products (Continuous)
# 5) DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
# 6) DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
# 7) CHANNEL: customersâ€™ Channel - Horeca (Hotel/Restaurant/CafÃ©) or Retail channel (Nominal)
# 8) REGION: customersâ€™ Region â€“ Lisnon, Oporto or Other (Nominal)
# Descriptive Statistics:
#
# 	(Minimum, Maximum, Mean, Std. Deviation)
# FRESH ( 3, 112151, 12000.30, 12647.329)
# MILK (55, 73498, 5796.27, 7380.377)
# GROCERY (3, 92780, 7951.28, 9503.163)
# FROZEN (25, 60869, 3071.93, 4854.673)
# DETERGENTS_PAPER (3, 40827, 2881.49, 4767.854)
# DELICATESSEN (3, 47943, 1524.87, 2820.106)
#
# REGION Frequency
# Lisbon 77
# Oporto 47
# Other Region 316
# Total 440
#
# CHANNEL Frequency
# Horeca 298
# Retail 142
# Total 440



fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module5/Datasets/WholesaleCustomersData.csv'
df = pd.read_csv(fname)
#df.my_feature.fillna( df.my_feature.mean() )
df.isnull().sum()
df.fillna(0)    # all zeros in numerical columns for NAs
df.drop(['Channel', 'Region'], 1, inplace=True)
print(df.describe())
df.plot.hist()

drop = {}
for col in df.columns:
    # Bottom 5
    sort = df.sort_values(by=col, ascending=True)
    if len(sort) > 5: sort=sort[:5]
    for index in sort.index: drop[index] = True # Just store the index once

    # Top 5
    sort = df.sort_values(by=col, ascending=False)
    if len(sort) > 5: sort=sort[:5]
    for index in sort.index: drop[index] = True # Just store the index once


print("Dropping {0} Outliers...".format(len(drop)))
df.drop(inplace=True, labels=drop.keys(), axis=0)
df.describe()




#
# Normalize:
#
# Let's say your user spend a LOT. Normalization divides each item by the average overall amount of spending.
# Stated differently, your new feature is = the contribution of overall spending going into that particular
# item: $spent on feature / $overall spent by sample.
#
#
# MinMax:
#
# What % in the overall range of $spent by all users on THIS particular feature is the current sample's
# feature at? When you're dealing with all the same units, this will produce a near face-value amount. Be
# careful though: if you have even a single outlier, it can cause all your data to get squashed up in lower percentages.
#
# Imagine your buyers usually spend $100 on wholesale milk, but today only spent $20. This is the relationship
# you're trying to capture with MinMax. NOTE: MinMax doesn't standardize (std. dev.); it only normalizes / unitizes
# your feature, in the mathematical sense. MinMax can be used as an alternative to zero mean, unit variance scaling.
# [(sampleFeatureValue-min) / (max-min)] * (max-min) + min Where min and max are for the overall feature values
# for all samples.

#T = preprocessing.StandardScaler().fit_transform(df)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
T = preprocessing.Normalizer().fit_transform(df)
#T = df # No Change


# Sometimes people perform PCA before doing KMeans, so that KMeans only operates on the most meaningful features.
# In our case, there are so few features that doing PCA ahead of time isn't really necessary, and you can do KMeans
# in feature space. But keep in mind you have the option to transform your data to bring down its dimensionality.
# If you take that route, then your Clusters will already be in PCA-transformed feature space, and you won't have to
# project them again for visualization.

# Sometimes people perform PCA before doing KMeans, so that KMeans only operates on the most meaningful features.
# In our case, there are so few features that doing PCA ahead of time isn't really necessary, and you can do KMeans
# in feature space. But keep in mind you have the option to transform your data to bring down its dimensionality.
# If you take that route, then your Clusters will already be in PCA-transformed feature space, and you won't have
# to project them again for visualization.

# Do KMeans

n_clusters = 3
centroids, labels = doKMeans(T, n_clusters)

print(centroids)

# Now do the PCA
display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)

# Visualize it:
fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
    # Plot the index of the sample, so you can further investigate it in your dset
    for i in range(len(T)): ax.text(T[i,0], T[i,1], df.index[i], color=c[labels[i]], alpha=0.75, zorder=600000)
    ax.set_xlim(min(T[:,0])*1.2, max(T[:,0])*1.2)
    ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
    # Plot a regular scatter plot
    sample_colors = [ c[labels[i]] for i in range(len(T)) ]
    ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)

# plot the centroids as X's and label them:
ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
for i in range(len(centroids)):
    ax.text(CC[i, 0], CC[i, 1], str(i), zorder=500010, fontsize=18, color=c[i])


# Display feature vectors for investigation:
if PLOT_VECTORS:
    drawVectors(T, display_pca.components_, df.columns, plt)


# Add the cluster label back into the dataframe and display it:
df['label'] = pd.Series(labels, index=df.index)
df

plt.show()















# ------------------------- misc code ----------------------
df.dropna(axis=0, how='any', inplace=True)
print(df.dtypes)
df['CallTime'] = pd.to_datetime(df['CallTime'])
InList = df['In'].unique()
user1 = df.loc[ df.In == InList[0] ]
user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
plt.show()
user1weekend = user1.loc[(user1.DOW == 'Sat') | (user1.DOW == 'Sun')]
user1weekend.reset_index()
user1weekendnight = user1weekend.loc[(user1weekend.CallTime < "06:00:00") | (user1weekend.CallTime > "22:00:00")]
# user1weekendnight.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user1weekendnight.TowerLon,user1weekendnight.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekend Calls (<6am or >10p)')
plt.show()
user1weekendnight = user1weekendnight[['TowerLon', 'TowerLat']]
from sklearn.cluster import KMeans
model = KMeans(n_clusters=1, init='random', n_init=100)
labels = model.fit_predict(user1weekendnight)

# Now we can print and plot the centroids:
centroids = model.cluster_centers_
print(centroids)
ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='blue', alpha=0.5, linewidths=3, s=169)
plt.show()


from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, init='random', n_init=100)
labels = model.fit_predict(user1weekendnight)

# Now we can print and plot the centroids:
centroids = model.cluster_centers_
print(centroids)
ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='blue', alpha=0.5, linewidths=3, s=169)
plt.show()








