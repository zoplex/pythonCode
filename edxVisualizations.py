#
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#
# matplotlib.style.use('ggplot')  # Look Pretty
# If the above line throws an error, use plt.style.use('ggplot') instead

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/ProgrammingwithPythonforDataScienceDAT210x'+ \
                 '/DAT210x-master/Module3/Datasets/students.data'
sds = pd.read_csv(fname)

sds.G3.plot.hist(title="first Python plot", bins=20)
plt.show()

# G3 feature by age group histograms:
#
sds[sds.age==18].G3.plot.hist(title="first Python plot", bins=20, alpha=0.5)
sds[sds.age==19].G3.plot.hist(title="first Python plot", bins=20, alpha=0.5)
sds[sds.age==16].G3.plot.hist(title="first Python plot", bins=20, alpha=0.5)
plt.show()


sds.plot.scatter(x='G1', y='G3')    # plot the whole df with x and y features

my_series = sds.G3
my_dataframe = sds[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)

#
#   3D scatter plots
#
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook   # for notebook interactive features
# %matplotlib inline
# %matplotlib gtk

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(sds.G1, sds.G3, sds['Dalc'], c='r', marker='.')
plt.show()



# parallel coordinates

from sklearn.datasets import load_iris
from pandas.tools.plotting import parallel_coordinates

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')
# If the above line throws an error, use plt.style.use('ggplot') instead

# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

df['target_names'] = [data.target_names[i] for i in data.target]

# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(df, 'target_names')
plt.show()

# andrew's curves
from sklearn.datasets import load_iris
from pandas.tools.plotting import andrews_curves

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')
# If the above line throws an error, use plt.style.use('ggplot') instead

# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target_names'] = [data.target_names[i] for i in data.target]

# Andrews Curves Start Here:
plt.figure()
andrews_curves(df, 'target_names')
plt.show()



#
#   imshow
#
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(1)
df = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
df.corr()

plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)

plt.show()

#
# labs
#
from matplotlib.pyplot import hist
fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module3/Datasets/wheat.data'
sds = pd.read_csv(fname, index_col=0)
sds = sds.dropna(axis=0, thresh=1)
sds2 = sds[['area', 'asymmetry']]
sds3 = sds[['groove', 'asymmetry']]
sds3 = sds3[sds3['groove'].notnull()]
sds3 = sds3[sds3['asymmetry'].notnull()]
sds2 = sds2[sds2['area'].notnull()]
sds2 = sds2[sds2['asymmetry'].notnull()]

hist(sds2['area'], weights=sds2['asymmetry'], alpha=0.75)
hist(sds3['groove'], weights=sds3['asymmetry'], alpha=0.75)
sds2.var()
sds3.var()

#
sds.plot.scatter(x='area', y='perimeter')
sds.plot.scatter(x='groove', y='asymmetry')
sds.plot.scatter(x='compactness', y='width')

# 3 D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')

ax.scatter(sds.area, sds.perimeter, sds['asymmetry'], c='r', marker='.')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('width')
ax.set_ylabel('groove')
ax.set_zlabel('length')

ax.scatter(sds.width, sds.groove, sds['length'], c='r', marker='.')
plt.show()
sds[ sds['length'] < 3]
# 4
sds = sds.drop(['id'], 1)      # drop 3 columns

plt.figure()
parallel_coordinates(sds, 'wheat_type')
plt.show()

sdsx = sds[ sds['wheat_type'] == 'rosa']
plt.figure()
parallel_coordinates(sdsx, 'wheat_type')
plt.show()

sdsx2 = sds[ sds['wheat_type'] == 'kama']
plt.figure()
parallel_coordinates(sdsx2, 'wheat_type')
plt.show()

sdsx3 = sds[ sds['wheat_type'] == 'canadian']
plt.figure()
parallel_coordinates(sdsx3, 'wheat_type')
plt.show()

# Andrews Curves Start Here:
plt.figure()
andrews_curves(sds, 'wheat_type')
plt.show()

plt.figure()
andrews_curves(sds[ sds['wheat_type'] == 'rosa'], 'wheat_type')
plt.show()


plt.figure()
andrews_curves(sds[ sds['wheat_type'] == 'canadian'], 'wheat_type')
plt.show()

plt.figure()
andrews_curves(sds[ sds['wheat_type'] == 'kama'], 'wheat_type')
plt.show()




#

#   imshow
#


from matplotlib.pyplot import hist
fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module3/Datasets/wheat.data'
sds = pd.read_csv(fname, index_col=0)
random.seed(1)
sds.corr()

plt.imshow(sds.corr(), cmap=plt.cm.Blues, interpolation='nearest')


def corrank(X):
    import itertools
    df = pd.DataFrame([[(i, j), X.corr().loc[i, j]] for i, j in list(itertools.combinations(X.corr(), 2))],
                      columns=['pairs', 'corr'])
    print(df.sort_values(by='corr', ascending=False))


corrank(sds)   # correlation sorted column pairs

sdsz = sds.drop(['area'], 1)      # drop 3 columns
sdsz = sdsz.drop(['perimeter'], 1)      # drop 3 columns
corrank(sdsz)
plt.imshow(sdsz.corr(), cmap=plt.cm.Blues, interpolation='nearest')



