import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_test, X_train, Y_test, Y_train = train_test_split(X, y, test_size=0.8, random_state=4)
print(X_train.shape)
print(X_test.shape)

# now try validation with differrent values of k
class NearestNeighborK(object):
      def __init__(self):
        pass

      def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

      def predict(self, X, k):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
          # find the nearest training image to the i'th test image
          # using the L1 distance (sum of absolute value differences)
          distancesAll = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
          distanceAll = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
          idx = np.argpartition(distancesAll,k)                                 # indexes of closest k elements
          closestYlabels = self.ytr[idx[:k]]                                    # labels of those indexes==closest k elements
          klist = list(closestYlabels)
          maxlabel = max(set(klist), key=klist.count)         # most common label
          Ypred[i] = maxlabel # predict the label of the nearest example

        return Ypred




validation_accuracies = []
X_val = X_train[:(int(X_train.shape[0]*0.1)), :]
Y_val = Y_train[:(int(Y_train.shape[0]*0.1))]
X_val.shape
Y_val.shape
for k in [1, 3, 5, 10, 20, 50, 100]:
        nn = NearestNeighborK()
        nn.train(X_train, Y_train)
        Y_predict = nn.predict(X_val, k = k)
        dflabel = np.column_stack((Y_val, Y_predict))
        accr = np.round(np.mean(Y_predict == Y_val) * 100.0, 2)
        print("accuracy achieved is %f percent" % accr)
        validation_accuracies.append((k, accr))

print("validation accuracies: ", validation_accuracies)



# done


