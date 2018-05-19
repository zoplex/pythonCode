# ********************************
# module 4-Lab2
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
pd.set_option('display.width', 300)
plt.style.use('ggplot')


def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)

    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled


def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax


# Do * NOT * alter this line, until instructed!
#scaleFeatures = False
scaleFeatures = True

fname = 'C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+ \
                    'ProgrammingwithPythonforDataScienceDAT210x/'+ \
                    'DAT210x-master/Module4/Datasets/kidney_disease.csv'
df = pd.read_csv(fname, index_col=0)

#df = df.drop(['wheat_type'], 1)
#
# We use the following representation to collect the dataset
#  age - age
# 	bp - blood pressure
# 	sg - specific gravity
# 	al - albumin
# 	su - sugar
# 	rbc - red blood cells
# 	pc - pus cell
# 	pcc - pus cell clumps
# 	ba - bacteria
# 	*** bgr - blood glucose random  ********************
#                   10. Blood Glucose Random(numerical)
#                       bgr in mgs/dl
#
# 	bu - blood urea
# 	sc - serum creatinine
# 	sod - sodium
# 	pot - potassium
# 	hemo - hemoglobin
# 	pcv - packed cell volume
# 	**** wc - white blood cell count  ********************
#                   17.     White Blood Cell Count(numerical)
# 	                        wc in cells/cumm
# 	**** rc - red blood cell count  ********************
#                   18.     Red Blood Cell Count(numerical)
# 	                        rc in millions/cmm
# 	htn - hypertension
# 	dm - diabetes mellitus
# 	cad - coronary artery disease
# 	appet - appetite
# 	pe - pedal edema
# 	ane - anemia
# 	class - class
#



df = df.dropna(axis=0)
df.reset_index()
df.isnull().sum()
df.dtypes
df.describe()
df.pcv = df.pcv.astype(int)
df.wc = df.wc.astype(int)
df.rc = df.rc.astype(float)
labels = ['red' if i=='ckd' else 'green' for i in df.classification]
df.drop(['classification', 'rbc', 'pc', 'pcc', 'ba'
                    , 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], 1, inplace=True)

df3 = df
df3.isnull().sum()

df3.describe()
df3.var().sort_values(ascending=False)

if scaleFeatures: df3 = scaleFeaturesDF(df3)

pca = PCA(n_components=2, svd_solver='full')
pca.fit(df3)
PCA(copy=True, n_components=2, whiten=False)
T = pca.transform(df3)
df3.shape
(430, 6) # 430 Student survey responses, 6 questions..
T.shape
(430, 2) # 430 Student survey responses, 2 principal components..

# Since we transformed via PCA, we no longer have column names; but we know we
# are in `principal-component` space, so we'll just define the coordinates accordingly:
ax = drawVectors(T, pca.components_, df3.columns.values, plt, scaleFeatures)
T  = pd.DataFrame(T)

T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)

plt.show()


