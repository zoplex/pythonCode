#
# KNN
#
from sklearn.neighbors import KNeighborsClassifier



# y = X['classification'].copy()
# X.drop(labels=['classification'], inplace=True, axis=1)
#
#
# X_train = pd.DataFrame([ [0], [1], [2], [3] ])
# y_train = [0, 0, 1, 1]
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, y_train)
# model.predict([[1.1]])
# model.predict_proba([[0.9]])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty

def plotDecisionBoundary(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.6
    resolution = 0.0025
    colors = ['royalblue','forestgreen','ghostwhite']

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
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

    # Plot the test original points as well...
    for label in range(len(np.unique(y))):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

    p = model.get_params()
    plt.axis('tight')
    plt.title('K = ' + str(p['n_neighbors']))


fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module3/Datasets/wheat.data'
X = pd.read_csv(fname, index_col=0)
Y = X['wheat_type'].copy()
X.drop(['wheat_type'], inplace=True, axis=1)
Y = Y.astype('category').cat.codes
Y.isnull().sum()        #  no nas
X.isnull().sum()
X.compactness.fillna(X.compactness.mean(), inplace=True)
X.width.fillna(X.width.mean(), inplace=True)
X.groove.fillna(X.groove.mean(), inplace=True)
X.isnull().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

from sklearn.preprocessing import Normalizer
norml = Normalizer()
norml.fit(X_train)
# we got trained normalizer here - now using it transform both train and test data
X_trainNrm = norml.transform(X_train)
X_testNrm = norml.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='randomized')
pcx = pca.fit(X_trainNrm)

train_pca = pca.transform(X_trainNrm)
test_pca  = pca.transform(X_testNrm)


from sklearn.neighbors import KNeighborsClassifier
modelknn = KNeighborsClassifier(n_neighbors=9)
modelknn.fit(train_pca, Y_train)

plotDecisionBoundary(modelknn, train_pca, Y_train)
print(modelknn.score(test_pca, Y_test))

# OR:
predictions = modelknn.predict(test_pca)
from sklearn.metrics import accuracy_score
acr = accuracy_score(Y_test, predictions)
print(acr)


for jj in range(9,0,-1):
    modelknn = KNeighborsClassifier(n_neighbors=jj)
    modelknn.fit(train_pca, Y_train)

    plotDecisionBoundary(modelknn, train_pca, Y_train)
    print(modelknn.score(test_pca, Y_test))