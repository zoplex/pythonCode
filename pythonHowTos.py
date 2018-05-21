pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)



# drop nas in df
df = df.dropna(how='any')
X.dropna(how='any', inplace=True, axis=0)


# read the file, skip index col
fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/kidney_disease.csv'
df = pd.read_csv(fname, index_col=0)
df = df.drop(['wheat_type'], 1)                 # drop column
# select some columns
df3 = df[['bgr', 'wc', 'rc']]
# drop rows with NAs
df = df.dropna(axis=0, thresh=1)
df.isnull().sum()                               # check nulls
X[pd.isnull(X).any(axis=1)]                     # another way

# examine data set

df.dtypes
df.describe()
df.corr()


# transform:
df[‘height2’] = pd.to_numeric(df.height, errors=’coerce’)
df.compactness.fillna( df.compactness.mean(), inplace=True)         # fill missing column values with its mean
df3.var().sort_values(ascending=False)                              # get varaince of all columns and sort by it, highest first

# drop nas-rows:
df = df.dropna(axis=0)
# reset indexes
df.reset_index()
# convert to types
df.pcv = df.pcv.astype(int)
df.wc = df.wc.astype(int)
df.rc = df.rc.astype(float)
# string type shows as object type
# check how many nas
df.isnull().sum()


# simple scatter plot:
ax.scatter(armadillo.x, armadillo.y, armadillo.z, c='green', marker='.', alpha=0.75)



# misc single liners:
labels = ['red' if i=='ckd' else 'green' for i in df.classification]
savdf.iloc[0:5, ]
savdf[1:4]['Duration']


# round up time to hour - up or down whichever is closest
savdf['callh'] = savdf['CallTime'].apply(lambda x: x.hour)
savdf['callmin'] = savdf['CallTime'].apply(lambda x: x.minute)
savdf.loc[ savdf['callmin'] > 30, 'callh']  +=1
savdf.loc[labels==0, 'callh'].mean()    # 8 hour!

# different nomralizers
#T = preprocessing.StandardScaler().fit_transform(df)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
T = preprocessing.Normalizer().fit_transform(df)
#T = df # No Change

# set column names / colnames:
df.columns = ['a', 'b']


# transforms

norm = Normalizer().fit(X_train)
X_Train = norm.transform(X_train)
X_test = norm.transform(X_test)

# stds = StandardScaler().fit(X_train)
# X_Train = stds.transform(X_train)
# X_test = stds.transform(X_test)
#
# minm = StandardScaler().fit(X_train)
# X_Train = minm.transform(X_train)
# X_test = minm.transform(X_test)
#
# robs = StandardScaler().fit(X_train)
# X_Train = robs.transform(X_train)
# X_test = robs.transform(X_test)

# set wide display in console:
pd.set_option('display.width', 300)

# convert to dummies:
X = pd.get_dummies(X)

# split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)



import pandas as pd
df1 = pd.read_csv('/home/zorankrunic0/PycharmProjects/DAT210x/Cached Datasets/dataset-har-PUC-Rio-ugulino/dataset-har-PUC-Rio-ugulino.csv'
                  ,sep=";")

# Pandas will check your dataset and then on a per-column basis decide if it's a numeric-type: int32,
# float32, float64. date-type: datetime64, timedelta[ns]. Or other object-type: object (string), category.
# If Pandas incorrectly assigns a type to a column, you can convert it, and we'll discuss that shortly.

pd.unique(df1.age)
df1[df1.y1==92].shape
df1[df1['y1'].isin([100,98])].shape
#The .loc[] method selects by column label, and .iloc[] selects by column index



import pandas as pd
dfm = pd.read_csv('/home/zorankrunic0/PycharmProjects/DAT210x/Module2/Datasets/direct_marketing.csv',sep=",")
print(dfm.dtypes)
print(df.shape)
print(dfm.describe())
dfm.head(5)

type(dfm[['zip_code']])
#Out[93]: pandas.core.frame.DataFrame
type(dfm['zip_code'])
#Out[94]: pandas.core.series.Series




