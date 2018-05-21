import numpy as np
import pandas as pd

x = np.zeros(100)

z2dar = np.array([[10,20,30], [50,60,70]])
isample = np.array([[12,13,14,12,11], [7,8,9,11,12]])

df1 = pd.DataFrame(columns=['c1','c2','c3'])
df1['c1'] = [10,20]
df1['c2'] = [11,12]
df1['c3'] = [12,13]

df2 = pd.DataFrame(columns=['c1','c2','c3'])
df2['c1'] = [11,20]
df2['c2'] = [13,12]
df2['c3'] = [15,13]




distances = np.sum(np.abs(df1 - df2.iloc[1,:]), axis = 1)


