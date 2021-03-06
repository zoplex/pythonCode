
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

pd.set_option('display.width', 300)

#   preprocessed by SubsetChanges4Python.R code to reduce the data to recent ones with 200k records so
#   it could be ingested here, and to fix header unreadable by Python:
X = pd.read_csv("C:/Users/zkrunic/Documents/BigData/ML/DSU/DSU-ML-2018/IncidentsExtractManual20180322csv.csv")




s = time.time()
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=0)
rfmodel.fit(X_train, y_train)
print(" completed rf fit in ", time.time()-s)
score = rfmodel.oob_score_
print("OOB Score: ", round(score*100, 3))
preds = rfmodel.predict(X_test)
print(pd.crosstab(y_test, preds))

print("Scoring...")
s = time.time()
score = rfmodel.score(X_test, y_test)
print("Score: ", round(score*100, 3))
print("Scoring completed in: ", time.time() - s)
list(zip(X_train, rfmodel.feature_importances_))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = pd.DataFrame(confusion_matrix(y_test, preds)
                  , columns=['actual no P1/P2s', 'actual P1/P2']
                  , index=['       pred no P1/P2', '       pred P1/P2'])
sns.heatmap(cm, annot=True, fmt='g')


