import pandas as pd

import matplotlib.pyplot as plt
from sklearn import svm


def load(path_train, path_test):
    # Load up the data.

    # You probably could have written this easily:
    with open(path_test, 'r')  as f: testing = pd.read_csv(f)
    with open(path_train, 'r') as f: training = pd.read_csv(f)

    # The number of samples between training and testing can vary
    # But the number of features better remain the same!
    n_features = testing.shape[1]

    X_test = testing.ix[:, :n_features - 1]
    X_train = training.ix[:, :n_features - 1]
    y_test = testing.ix[:, n_features - 1:].values.ravel()
    y_train = training.ix[:, n_features - 1:].values.ravel()

    # Special:
    # ...

    return X_train, X_test, y_train, y_test


def peekData(X_train):
    # The 'targets' or labels are stored in y. The 'samples' or data is stored in X
    print("Peeking your data...")
    fig = plt.figure()
    fig.set_tight_layout(True)

    cnt = 0
    for col in range(5):
        for row in range(10):
            plt.subplot(5, 10, cnt + 1)
            plt.imshow(X_train.ix[cnt, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.axis('off')
            cnt += 1

    plt.show()


def drawPredictions(X_train, X_test, y_train, y_test):
    fig = plt.figure()
    fig.set_tight_layout(True)

    # Make some guesses
    y_guess = model.predict(X_test)

    # INFO: This is the second lab we're demonstrating how to
    # do multi-plots using matplot lab. In the next assignment(s),
    # it'll be your responsibility to use this and assignment #1
    # as tutorials to add in the plotting code yourself!
    num_rows = 10
    num_cols = 5

    index = 0
    for col in range(num_cols):
        for row in range(num_rows):
            plt.subplot(num_cols, num_rows, index + 1)

            # 8x8 is the size of the image, 64 pixels
            plt.imshow(X_test.ix[index, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')

            # Green = Guessed right
            # Red = Fail!
            fontcolor = 'g' if y_test[index] == y_guess[index] else 'r'
            plt.title('Label: %i' % y_guess[index], fontsize=6, color=fontcolor)
            plt.axis('off')
            index += 1
    plt.show()

fname1 = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/optdigits.tes'
fname2 = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/optdigits.tra'

X_train, X_test, y_train, y_test = load(fname1, fname2)
import matplotlib.pyplot as plt
from sklearn import svm

peekData(X_train)

from sklearn.svm import SVC
model = SVC(C=1, gamma=0.001, kernel='linear')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score:\n", score)
drawPredictions( X_train, X_test, y_train, y_test)

true_1000th_test_value = y_test[1000]
print("1000th test label: ", true_1000th_test_value)
y_guess = model.predict(X_test)
guess_1000th_test_value = y_guess[1000]
print("1000th test prediction: ", guess_1000th_test_value)

onex = X_test.iloc[1000:1001]
guess_1000th_test_value =  model.predict(onex)
print("1000th test prediction: ", guess_1000th_test_value)

pd.set_option('display.width', 300)
plt.imshow(X_test.ix[1000, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


from sklearn.svm import SVC
model = SVC(C=1, gamma=0.001, kernel='poly')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score:\n", score)
drawPredictions( X_train, X_test, y_train, y_test)

true_1000th_test_value = y_test[1000]
print("1000th test label: ", true_1000th_test_value)
y_guess = model.predict(X_test)
guess_1000th_test_value = y_guess[1000]
print("1000th test prediction: ", guess_1000th_test_value)

onex = X_test.iloc[1000:1001]
guess_1000th_test_value =  model.predict(onex)
print("1000th test prediction: ", guess_1000th_test_value)

pd.set_option('display.width', 300)
plt.imshow(X_test.ix[1000, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


from sklearn.svm import SVC
model = SVC(C=4, gamma=0.001, kernel='rbf')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score:\n", score)
#drawPredictions( X_train, X_test, y_train, y_test)

true_1000th_test_value = y_test[1000]
print("1000th test label: ", true_1000th_test_value)
y_guess = model.predict(X_test)
guess_1000th_test_value = y_guess[1000]
print("1000th test prediction: ", guess_1000th_test_value)

onex = X_test.iloc[1000:1001]
guess_1000th_test_value =  model.predict(onex)
print("1000th test prediction: ", guess_1000th_test_value)

pd.set_option('display.width', 300)
plt.imshow(X_test.ix[1000, :].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

from sklearn.svm import SVC
for kk in range(1,10):
    gamma = 0.0013+(kk*0.00003)
    model = SVC(C=3, gamma=gamma, kernel='rbf')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Score for C=2 and gamma=", gamma, " is :\n", score)