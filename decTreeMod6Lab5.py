import pandas as pd



fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module6/Datasets/agaricus-lepiota.data'
#X = pd.read_csv(fname, index_col=0)
X = pd.read_csv(fname, na_values='?')
X[pd.isnull(X).any(axis=1)]
clms = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
    'gill_attach', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
    'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'viel_type',
    'viel_color', 'ring_number', 'ring_type', 'spore_print_color',
    'population', 'habitat']
X.columns = clms
X.isnull().sum()
X.dropna(axis=0, inplace=True, how='any')
y = X['class']
X.drop(labels=['class'], axis=1, inplace=True)
y = y.map({'e': 0, 'p': 1})


X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

from sklearn import tree
dect = tree.DecisionTreeClassifier()
dect.fit(X_train, y_train)
score = dect.score(X_test, y_test)
print("High-Dimensionality Score: ", round((score*100), 3))

print(dect.feature_importances_)
names = X_train.columns
print(sorted(zip(map(lambda x: round(x, 4), dect.feature_importances_), names),
             reverse=True))




