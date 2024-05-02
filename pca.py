import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.svm import SVR

df = pd.read_csv('XXX.csv', sep=";" )

#removing variables 
df.drop(['Durulama_Sayisi'], axis=1, inplace=True)

# Split into train and test
CAO = df.iloc[:,-1].values
df  = df.iloc[:,0:40]
print(CAO) 

Cikis_AO = pd.DataFrame(data = CAO, index= range(180), columns = ['CAO'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df,Cikis_AO,
                                                    test_size=0.40, 
                                                   random_state=0)

# Standardization
sc= StandardScaler()

x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test  = sc.fit_transform(y_test)

# PCA

pca = PCA(n_components=20)

x_train_2 = pca.fit_transform(x_train)
x_test_2  = pca.transform(x_test)

# Before PCA 

svr_reg = SVR()
svr_reg.fit(x_train, y_train)
tahmin = svr_reg.predict(x_test)

# evaluation of the model with r2

r2_score(y_test, tahmin)
# 0.7878388273269488
plt.scatter(y_test, tahmin, color='red')

# After PCA

svr_reg2 = SVR()
svr_reg2.fit(x_train_2, y_train)
tahmin2 = svr_reg2.predict(x_test_2)

# Evaluation of the model with r2

r2_score(y_test, tahmin2)
#  0.7957258534317097
plt.scatter(y_test, tahmin2, color='red')

