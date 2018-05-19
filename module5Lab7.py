#
#
#   module5/lab7
#
import random, math
import pandas as pd
import numpy as np
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


# Leave this alone until indicated:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
    print("Plotting...")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.1
    resolution = 0.1

    #(2 for benign, 4 for malignant)
    colors = {2:'royalblue', 4:'lightsalmon'}


    # Calculate the boundaris
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
    plt.axis('tight')

    # Plot your testing points as well...
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

    p = model.get_params()
    plt.title('K = ' + str(p['n_neighbors']))
    plt.show()



pd.set_option('display.width', 300)

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module5/Datasets/breast-cancer-wisconsin.data'
df = pd.read_csv(fname, header=None, na_values='?')
df.columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial'
    , 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']

y = df['status'].copy()
df.drop(labels=['sample', 'status'], inplace=True, axis=1)
df.isnull().sum()
df.nuclei.fillna(df.nuclei.mean(), inplace=True)
df.isnull().sum()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.5, random_state=7)

from sklearn.preprocessing import Normalizer, MinMaxScaler, RobustScaler, StandardScaler



# norm = Normalizer().fit(X_train)
# X_train = norm.transform(X_train)
# X_test = norm.transform(X_test)

stds = StandardScaler().fit(X_train)
X_train = stds.transform(X_train)
X_test = stds.transform(X_test)
#
# minm = MinMaxScaler().fit(X_train)
# X_train = minm.transform(X_train)
# X_test = minm.transform(X_test)
#
# robs = RobustScaler().fit(X_train)
# X_train = robs.transform(X_train)
# X_test = robs.transform(X_test)


model = None

if Test_PCA:
    print('Computing 2D Principle Components')
    # TODO: Implement PCA here. Save your model into the variable 'model'.
    # You should reduce down to two dimensions.

    # .. your code here ..
    from sklearn.decomposition import PCA

    model = PCA(n_components=2)
    model.fit(X_train)

else:
    print('Computing 2D Isomap Manifold')
    # TODO: Implement Isomap here. Save your model into the variable 'model'
    # Experiment with K values from 5-10.
    # You should reduce down to two dimensions.

    # .. your code here ..
    from sklearn.manifold import Isomap

    model = Isomap(n_neighbors=5, n_components=2)
    model.fit(X_train)



X_train = model.transform(X_train)
X_test = model.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
# modelknn = KNeighborsClassifier(n_neighbors=15, weights='distance')
modelknn = KNeighborsClassifier(n_neighbors=15, weights='uniform')
modelknn.fit(X_train, y_train)

accr = modelknn.score(X_test, y_test)
print(accr)

plotDecisionBoundary(modelknn, X_test, y_test)


