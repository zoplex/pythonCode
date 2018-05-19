#
#  transformation
#


#
#   PCA - unsupervised dimensionality reduction alg.
#
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
import pandas as pd

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module3/Datasets/wheat.data'
df = pd.read_csv(fname, index_col=0)
df = df.drop(['wheat_type'], 1)
df = df.dropna(axis=0, thresh=1)
df.isnull().sum()
#
df.compactness.fillna( df.compactness.mean(), inplace=True)
df.width.fillna( df.width.mean(), inplace=True)
df.groove.fillna( df.groove.mean(), inplace=True)
df.isnull().sum()

pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)
T = pca.transform(df)
df.shape
(430, 6) # 430 Student survey responses, 6 questions..
T.shape
(430, 2) # 430 Student survey responses, 2 principal components..



# *************** ARMADILO  plot ********************************
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time

from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement

plt.style.use('ggplot')
reduce_factor = 100

plyfile = PlyData.read('C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/stanford_armadillo.ply')

armadillo = pd.DataFrame({
  'x':plyfile['vertex']['z'][::reduce_factor],
  'y':plyfile['vertex']['x'][::reduce_factor],
  'z':plyfile['vertex']['y'][::reduce_factor]
})


def do_PCA(armadillo, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver)
    pca.fit(armadillo)
    PCA(copy=True, n_components=2, whiten=False)
    T = pca.transform(armadillo)
    armadillo.shape
    T.shape
    return T




fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.set_title('Armadillo 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(armadillo.x, armadillo.y, armadillo.z, c='green', marker='.', alpha=0.75)


# Render the Original Armadillo

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.set_title('Armadillo 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(armadillo.x, armadillo.y, armadillo.z, c='green', marker='.', alpha=0.75)

# time the executioin:
start = time.time()
pca = do_PCA(armadillo, 'full')
end = time.time()
print(end-start)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Full PCA')
ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)
plt.show()



# try randomized
start = time.time()
rpca = do_PCA(armadillo, 'randomized')
end = time.time()
print(end-start)



