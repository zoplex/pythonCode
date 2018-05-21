import numpy as np
import pandas as pd

x = np.zeros(100)

z2dar = np.array([[10,20,30], [50,60,70]])
isample = np.array([[12,13,14,12,11], [7,8,9,11,12]])

df1 = pd.DataFrame(columns=['c1','c2','c3'])
df1['c1'] = [10,20]
df1['c2'] = [11,12]
df1['c3'] = [12,13]

df2 = pd.DataFrame(columns=['c1','c2','c3'])
df2['c1'] = [11,20]
df2['c2'] = [13,12]
df2['c3'] = [15,13]


distances = np.sum(np.abs(df1 - df2.iloc[1,:]), axis = 1)

import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred




