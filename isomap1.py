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
    for i in range(num_to_plot):
        img_num = int(random.random() * num_images)
        x0, y0 = T.loc[img_num, x] - x_size / 2., T.loc[img_num, y] - y_size / 2.
        x1, y1 = T.loc[img_num, x] + x_size / 2., T.loc[img_num, y] + y_size / 2.
        img = df.iloc[img_num, :].reshape(num_pixels, num_pixels)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # It also plots the full scatter:
    ax.scatter(T.loc[:, x], T.loc[:, y], marker='.', alpha=0.7)


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



fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/face_data.mat'
mat = scipy.io.loadmat(fname)
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
    df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)


scaleFeatures = True
if scaleFeatures: df3dmf = scaleFeaturesDF(df)

# ----
pca = PCA(n_components=3, svd_solver='full')
pca.fit(df)
PCA(copy=True, n_components=3, whiten=False)
T = pca.transform(df)
# df.shape
# (430, 6) # 430 Student survey responses, 6 questions..
# T.shape
# (430, 2) # 430 Student survey responses, 2 principal components..

# Since we transformed via PCA, we no longer have column names; but we know we
# are in `principal-component` space, so we'll just define the coordinates accordingly:
#ax = drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2', 'component3']
Plot2D(T, "PCA - 1-2", 'component1', 'component2', num_to_plot=40)
plt.show()

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=3, n_components=3)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=4
                , neighbors_algorithm='auto', path_method='auto', tol=0)

manifold = iso.transform(df)
mf  = pd.DataFrame(manifold)

mf.columns = ['component1', 'component2', 'component3']
Plot2D(mf, "MANIFOLDS - components 1-2", 'component1', 'component2', num_to_plot=40)

plt.show()

