
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time


fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/parkinsons.data'
X = pd.read_csv(fname, index_col=0)
y = X['status']
X.drop(['status'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("score is ", score)

best_score = 0.0
best_c = 0
best_gamma = 0.0
for i in np.arange(start=0.05, stop=2.05, step = 0.05):
    for j in np.arange(start = 0.001, stop = 0.1, step = 0.001):
        model = SVC(C=i, gamma=j)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if ( score > best_score):
            best_score = score
            best_c = model.C
            best_gamma = model.gamma
            print("score improved( C=", best_c, ", gamma=", best_gamma, ") to ", best_score)

print("BEST score ( C=", best_c, ", gamma=", best_gamma, ") to ", best_score)

# -------------------- do it again but with data transformation ---------------------------------
fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/parkinsons.data'
X = pd.read_csv(fname, index_col=0)
y = X['status']
X.drop(['status'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

from sklearn import preprocessing

# BEST score ( C= 0.05 , gamma= 0.001 ) to  0.7966101694915254
# norm = preprocessing.Normalizer().fit(X_train)
# X_train = norm.transform(X_train)
# X_test = norm.transform(X_test)

# BEST score ( C= 1.55 , gamma= 0.097 ) to  0.9322033898305084
# stds = preprocessing.StandardScaler().fit(X_train)
# X_train = stds.transform(X_train)
# X_test = stds.transform(X_test)
# #
# BEST score ( C= 0.7500000000000001 , gamma= 0.097 ) to  0.8813559322033898
# minm = preprocessing.MinMaxScaler().fit(X_train)
# X_train = minm.transform(X_train)
# X_test = minm.transform(X_test)

# BEST score ( C= 1.2000000000000002 , gamma= 0.098 ) to  0.8813559322033898
# maxa = preprocessing.MaxAbsScaler().fit(X_train)
# X_train = maxa.transform(X_train)
# X_test = maxa.transform(X_test)
#

# BEST score ( C= 1.85 , gamma= 0.067 ) to  0.9152542372881356
robs = preprocessing.RobustScaler().fit(X_train)
X_train = robs.transform(X_train)
X_test = robs.transform(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("score is ", score)

best_score = 0.0
best_c = 0
best_gamma = 0.0
for i in np.arange(start=0.05, stop=2.05, step = 0.05):
    for j in np.arange(start = 0.001, stop = 0.1, step = 0.001):
        model = SVC(C=i, gamma=j)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if ( score > best_score):
            best_score = score
            best_c = model.C
            best_gamma = model.gamma
            print("score improved( C=", best_c, ", gamma=", best_gamma, ") to ", best_score)

print("BEST score ( C=", best_c, ", gamma=", best_gamma, ") to ", best_score)


# -------------------- do it again but with data transformation ---------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import time


fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/parkinsons.data'
X = pd.read_csv(fname, index_col=0)
y = X['status']
X.drop(['status'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

from sklearn import preprocessing
from sklearn.manifold import Isomap

best_score = 0.0
best_c = 0
best_gamma = 0.0
best_kngh = 0
best_ncomp = 0
for kk in range(2, 6):
    for ii in range(4, 7):
        # BEST score ( C= 1.85 , gamma= 0.067 ) to  0.9152542372881356
        robs = preprocessing.RobustScaler().fit(X_train)
        X_train = robs.transform(X_train)
        X_test = robs.transform(X_test)

        iso = Isomap(n_neighbors=kk, n_components=ii)
        iso.fit(X_train)
        X_train = iso.transform(X_train)
        X_test = iso.transform(X_test)

        from sklearn.svm import SVC

        model = SVC()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # print("score is ", score)

        for i in np.arange(start=0.05, stop=2.05, step=0.05):
            for j in np.arange(start=0.001, stop=0.1, step=0.001):
                model = SVC(C=i, gamma=j)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                if (score > best_score):
                    best_score = score
                    best_c = model.C
                    best_gamma = model.gamma
                    best_kngh = kk
                    best_ncomp = ii
                    print("score improved( C=", best_c, ", gamma=", best_gamma
                          , ", neigh=", kk, ", ncomp=", ii, ") to ", best_score)

print("score improved( C=", best_c, ", gamma=", best_gamma
      , ", neigh=", kk, ", ncomp=", ii, ") to ", best_score)



