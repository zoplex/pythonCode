import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') # Look Pretty

#
def drawLine(model, X_test, y_test, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_test, y_test, c='g', marker='o')
    ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

    print("Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0])
    print("Est 2015 " + title + " Life Expectancy: ", model.predict([[2015]])[0])
    print("Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0])
    print("Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0])

    score = model.score(X_test, y_test)
    title += " R2: " + str(score)
    ax.set_title(title)

    plt.show()



pd.set_option('display.width', 300)

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module5/Datasets/life_expectancy.csv'
X = pd.read_csv(fname, delim_whitespace=True)

from sklearn import linear_model
model = linear_model.LinearRegression()

X_train = X.Year[X.Year < 1986]
y_train = X.WhiteMale[X.Year < 1986]
X_train = X_train.to_frame()
model.fit(X_train, y_train)
drawLine( model, X_train, y_train, "WhiteMale")
print("actual life expectancy 2014 from loaded ds: ", X.WhiteMale[ (X.Year==2014)].values[0])


X_train = X.Year[X.Year < 1986]
y_train = X.BlackFemale[X.Year < 1986]
X_train = X_train.to_frame()
model.fit(X_train, y_train)
drawLine( model, X_train, y_train, "BlackFemale")
print("actual life expectancy 2014 from loaded ds: ", X.BlackFemale[ (X.Year==2014)].values[0])


print(X.corr())
plt.imshow(X.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.show()


X_train = X.Year[X.Year < 1986]
y_train = X.WhiteMale[X.Year < 1986]
X_train = X_train.to_frame()
model.fit(X_train, y_train)
drawLine( model, X_train, y_train, "WhiteMale")


