import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot') # Look Pretty



def drawLine(model, X_test, y_test, title, R2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_test, y_test, c='g', marker='o')
    ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

    title += " R2: " + str(R2)
    ax.set_title(title)
    print(title)
    print("Intercept(s): ", model.intercept_)

    plt.show()


def drawPlane(model, X_test, y_test, title, R2):
    # This convenience method will take care of plotting your
    # test observations, comparing them to the regression plane,
    # and displaying the R2 coefficient
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlabel('prediction')

    # You might have passed in a DataFrame, a Series (slice),
    # an NDArray, or a Python List... so let's keep it simple:
    X_test = np.array(X_test)
    col1 = X_test[:, 0]
    col2 = X_test[:, 1]

    # Set up a Grid. We could have predicted on the actual
    # col1, col2 values directly; but that would have generated
    # a mesh with WAY too fine a grid, which would have detracted
    # from the visualization
    x_min, x_max = col1.min(), col1.max()
    y_min, y_max = col2.min(), col2.max()
    x = np.arange(x_min, x_max, (x_max - x_min) / 10)
    y = np.arange(y_min, y_max, (y_max - y_min) / 10)
    x, y = np.meshgrid(x, y)

    # Predict based on possible input values that span the domain
    # of the x and y inputs:
    z = model.predict(np.c_[x.ravel(), y.ravel()])
    z = z.reshape(x.shape)

    ax.scatter(col1, col2, y_test, c='g', marker='o')
    ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)

    title += " R2: " + str(R2)
    ax.set_title(title)
    print(title)
    print("Intercept(s): ", model.intercept_)

    plt.show()


pd.set_option('display.width', 300)

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module5/Datasets/college.csv'
X = pd.read_csv(fname, index_col=0)
print(X.head())
print(X.info())
print(X.describe())
print(X.dtypes)
print(X.isnull().sum())
print(X.columns)
print(X.shape)

X.Private = X.Private.map({'Yes': 1, 'No': 0})

from sklearn import linear_model
model = linear_model.LinearRegression()

# --------- acc = f(rb) Accept(Room&Board) R2: -0.0026669864145500983 -------------------------------
X_rb = X[['Room.Board']]
X_accept = X[['Accept']]
from sklearn.model_selection import train_test_split
model = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_rb, X_accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(Room&Board)", score)

# --------- acc = f(rb) R2: acc(enroll) 0.8578204867356156 -----------------------------

X_enroll = X[['Enroll']]
X_accept = X[['Accept']]
from sklearn.model_selection import train_test_split
model = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_enroll, X_accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(Room&Board)", score)


# --------- acc = f(rb) R2: Accept(F.Undergrad) R2: 0.7779917973754007 -----------------------------

X_ug = X[['F.Undergrad']]
X_accept = X[['Accept']]
from sklearn.model_selection import train_test_split
model = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_ug, X_accept, test_size=0.3, random_state=7)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(F.Undergrad)", score)




# --------- acc = f(rb) Accept(Room&Board) R2: -0.0026669864145500983 -------------------------------
X_rben = X[['Room.Board', 'Enroll']]
X_accept = X[['Accept']]
from sklearn.model_selection import train_test_split
model = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_rben, X_accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# next funciton call crashed on this example, but worked ok when expanded bellow ... leave it
# that way - no time for investigating now ...
#
#drawLine(model, X_test, y_test, "Accept(Room&Board+Enroll)", score)
#---
title="Accept(Room&Board+Enroll)"
fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlabel('prediction')

# You might have passed in a DataFrame, a Series (slice),
# an NDArray, or a Python List... so let's keep it simple:
X_test = np.array(X_test)
col1 = X_test[:, 0]
col2 = X_test[:, 1]

# Set up a Grid. We could have predicted on the actual
# col1, col2 values directly; but that would have generated
# a mesh with WAY too fine a grid, which would have detracted
# from the visualization
x_min, x_max = col1.min(), col1.max()
y_min, y_max = col2.min(), col2.max()
x = np.arange(x_min, x_max, (x_max - x_min) / 10)
y = np.arange(y_min, y_max, (y_max - y_min) / 10)
x, y = np.meshgrid(x, y)

# Predict based on possible input values that span the domain
# of the x and y inputs:
z = model.predict(np.c_[x.ravel(), y.ravel()])
z = z.reshape(x.shape)

ax.scatter(col1, col2, y_test, c='g', marker='o')
ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)

title += " R2: " + str(R2)
ax.set_title(title)
print(title)
print("Intercept(s): ", model.intercept_)

plt.show()

#


