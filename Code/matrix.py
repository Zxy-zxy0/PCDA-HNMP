import pandas as pd
import numpy as np

#cc
x = pd.read_csv('../RawData/Relevance/cList.csv')
y = pd.read_csv('../RawData/Relevance/cList.csv')
data = pd.read_csv('../RawData/Relevance/acc.csv')
valsx = x.index
valsy = y.index
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/circRNA-circRNA.txt', df, fmt='%d', delimiter='\t')

#cd
x = pd.read_csv('../RawData/Relevance/cList.csv')
y = pd.read_csv('../RawData/Relevance/dList.csv')
data = pd.read_csv('../RawData/Relevance/acd.csv')
valsx = x.index
valsy = y.index
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/circRNA-Disease.txt', df, fmt='%d', delimiter='\t')

#ci
x = pd.read_csv('../RawData/Relevance/cList.csv')
y = pd.read_csv('../RawData/Relevance/iList.csv')
data = pd.read_csv('../RawData/Relevance/aci.csv')
valsx = x.index
valsy = y.index
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/circRNA-miRNA.txt', df, fmt='%d', delimiter='\t')


#dd
x = pd.read_csv('../RawData/Relevance/dList.csv')
y = pd.read_csv('../RawData/Relevance/dList.csv')
data = pd.read_csv('../RawData/Relevance/add.csv')
valsx = x.index
valsy = y.index
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/Disease-Disease.txt', df, fmt='%d', delimiter='\t')

#di
x = pd.read_csv('../RawData/Relevance/dList.csv')
y = pd.read_csv('../RawData/Relevance/iList.csv')
data = pd.read_csv('../RawData/Relevance/adi.csv')
valsx = np.unique(x[['index']])
valsy = np.unique(y[['index']])
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/Disease-miRNA.txt', df, fmt='%d', delimiter='\t')

#im
x = pd.read_csv('../RawData/Relevance/iList.csv')
y = pd.read_csv('../RawData/Relevance/mList.csv')
data = pd.read_csv('../RawData/Relevance/aim.csv')
valsx = np.unique(x[['index']])
valsy = np.unique(y[['index']])
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/miRNA-mRNA.txt', df, fmt='%d', delimiter='\t')

#md
x = pd.read_csv('../RawData/Relevance/mList.csv')
y = pd.read_csv('../RawData/Relevance/dList.csv')
data = pd.read_csv('../RawData/Relevance/amd.csv')
valsx = np.unique(x[['index']])
valsy = np.unique(y[['index']])
df = pd.DataFrame(0, index=valsx, columns=valsy)
df.values[data.x, data.y] = 1
np.savetxt('../Data/matrix/mRNA-Disease.txt', df, fmt='%d', delimiter='\t')


