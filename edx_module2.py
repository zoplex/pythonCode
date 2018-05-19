#
import pandas as pd
pd.set_option('display.width', 300)
df = pd.read_csv('C:/Users/zkrunic/Documents/BigData/ML/Python/edx/ProgrammingwithPythonforDataScienceDAT210x'+
                 '/DAT210x-master/Module2/Datasets/tutorial.csv', sep=',')
df.describe()
df.loc[2:4, 'col3']





df = pd.read_csv('C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+
                 'ProgrammingwithPythonforDataScienceDAT210x/'+
                 'DAT210x-master/Module2/Datasets/servo.data', sep=',', header=None )
df.describe()

# Attribute Information:
#
# 1. motor: A,B,C,D,E
#  2. screw: A,B,C,D,E
#  3. pgain: 3,4,5,6
#  4. vgain: 1,2,3,4,5
#  5. class: 0.13 to 7.10

df.columns = ['motor', 'screw', 'pgain', 'vgain', 'class2']
ulist = df['motor'].unique()
ulist.sort()
print(ulist)

df5 = df.loc[ df.vgain==5, ['vgain']]
df.vgain.value_counts()

dfq2 = df.loc[ (df['motor'] == 'E') & (df['screw'] == 'E')]

zmean = df.loc[ df.pgain==4, ['vgain']].mean()
print(zmean)

# # read html page:
import pandas as pd
import html5lib
url = 'http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
df_list = pd.read_html(url)
df = df_list[-1]
print(df)
df = df.loc[2:, :]

# colnames: RK	PLAYER	TEAM	GP	G	A	PTS	+/-	PIM	PTS/G	SOG	PCT	GWG	G	A	G	A
df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', 'PdM', 'PIM'
    , 'PTSG', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']


df = df.dropna(axis=0, thresh=4)
df = df.drop(columns=['RK']).reset_index()
df = df.drop( df.index[[10, 21, 32]])   # drop repeating header rows
df[['GP', 'G', 'A', 'PTS', 'PdM', 'PIM', 'PTSG', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']] = df[[
    'GP', 'G', 'A', 'PTS', 'PdM', 'PIM', 'PTSG', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']].apply(pd.to_numeric)
df.shape
len(df.PCT.unique())
df.loc[[15, 16], ['GP']].sum()


#
import pandas as pd
import html5lib
df = pd.read_csv('C:/Users/zkrunic/Documents/BigData/ML/Python/edx/'+
                 'ProgrammingwithPythonforDataScienceDAT210x/'+
                 'DAT210x-master/Module2/Datasets/census.data', sep=',', header=None )

df.columns = ['nope', 'education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']
df.drop(columns=0)