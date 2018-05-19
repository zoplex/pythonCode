#
#   dataset-har-PUC-Rio-ugulino.csv
#
import pandas as pd
import time

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/dataset-har-PUC-Rio-ugulino.csv'
#X = pd.read_csv(fname, index_col=0)
X = pd.read_csv(fname, sep = ';', na_values='?', decimal=',')
X[pd.isnull(X).any(axis=1)]
X.isnull().sum()
X = X.dropna(axis=0)

X.z4 = pd.to_numeric(X.z4, errors='coerce')
X.gender = X.gender.map({'Woman': 1, 'Man': 0})
X['class'].unique()
y = X[['class']]
y = pd.get_dummies(y)
X.drop(labels=['class', 'user'], inplace=True, axis=1)
print(X.describe())
X.isnull().sum()

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=80, max_depth=10, oob_score=True, random_state=0)
model.fit(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

s = time.time()
model = rfmodel.fit(X_train, y_train)
print(" completed rf fit in ", time.time()-s)

score = model.oob_score_
print("OOB Score: ", round(score*100, 3))


print("Scoring...")
s = time.time()
# TODO: score your model on your test set
score = model.score(X_test, y_test)
print("Score: ", round(score*100, 3))
print("Scoring completed in: ", time.time() - s)



