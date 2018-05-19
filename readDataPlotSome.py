
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 300)

ch1k = pd.read_csv("C:/Users/zkrunic/Documents/BigData/ML/DSU/DSU-ML-2018/dataCommon/SM_CHANGES.csv", nrows=100000)
ch1k.describe()
ch1k.head()
list(ch1k)

ch1k = ch1k[np.isfinite(ch1k['Duration'])]
ch1k['Close Time'] = pd.to_datetime(ch1k['Close Time'])
ch1k['doweek'] = ch1k['Close Time'].dt.dayofweek
ch1k = ch1k.sort_values(by='Close Time')
ch1k = ch1k.loc[(ch1k['Close Time'] > '2015-01-01'), :]
ch1k = ch1k.loc[ch1k['Duration'] < 1000, :]
ch1k['Duration'] /= 60

dfplot = ch1k.groupby('doweek', as_index=False)['Duration'].mean()
plt.bar (dfplot['doweek'], dfplot['Duration'], color='orange')
plt.xlabel('change day of the week (close time)')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.ylabel('change duration (hours)')
plt.title("Average change tickets duration by date")
plt.show()





# #       affected CIs for the changes:
# ptask = pd.read_csv("C:/Users/zkrunic/Documents/BigData/ML/DSU/DSU-ML-2018/dataCommon/SM_PRBLEM_TASKS.csv", nrows=100000)
# print(ptask.describe())
# ptask.dtypes
# ptask.head()
# ptask.shape


# plt.scatter( DTAffCI[['Priority Code']], DTAffCI[['Sn Over 9 Months']])
# plt.hist( DTAffCI[['Sn Over 9 Months']], bins=5)
# plt.scatter(DTAffCI[['Priority Code']], DTAffCI[['Number of Records']])
# newcolnames     <- c("Ch.Id", colnames(DTAffCI)[2:ncol(DTAffCI)])
# DTAffCI         <- setNames(DTAffCI, newcolnames)
# #       list of unique CI IDs for all Change Tickets - total of all affected CIS for all changes
# uniqueChgAffCI  <- unique(DTAffCI$Ci.Ucmdb)
# #       from INC side
# uniqueIncAffCI  <- unique(DTINC$LOGICAL_NAME)
# #       list of Incident-related CIds for which there is/are some Changes - they overlap in CI IDs
# IncCIDwithChg   <- uniqueIncAffCI[uniqueIncAffCI %in% uniqueChgAffCI]
# #       now select those changes that are related to this Inc CI ID list
# #               -- first get Changes trhat have at least one affected CI from the INC list
# DTAffCIincRel   <- DTAffCI %>% filter( Ci.Ucmdb %in% IncCIDwithChg )    # about 10% of the total changes
# #       unique lidt of changes for that x-ref
# uniqueChgIncRel <- unique(DTAffCIincRel$Ch.Id)                          # about $180k
# #       now get original change tickets for this subset
# DT_Chg_Inc      <- DT_Chg %>% filter( Ch.Id %in% uniqueChgIncRel )      # 180k out of original 5 mil changes
# save( DT_Chg_Inc, file="../../dataCommon/DT_Chg_Inc.dat")
# cat("all done\n")
# Sys.time()
