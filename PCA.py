# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:29:59 2020

@author: DSİ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sns

df = pd.read_csv('C:\\Users\\DSİ\\Desktop\\ar_proj_.csv', sep=";" )

#removing variables 
df.drop(['Durulama_Sayisi'], axis=1, inplace=True)

#♠veriyi train ve test olarak bölme
CAO = df.iloc[:,-1].values
df  = df.iloc[:,0:40]
print(CAO) 

Cikis_AO = pd.DataFrame(data = CAO, index= range(180), columns = ['CAO'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df,Cikis_AO,
                                                    test_size=0.40, 
                                                   random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
# standardization
#feature scaling-standardisation
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test  = sc.fit_transform(y_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=20)

x_train_2 = pca.fit_transform(x_train)
x_test_2  = pca.transform(x_test)

# pca dönüşümünden önce gelen SVR
from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(x_train, y_train)
tahmin = svr_reg.predict(x_test)

#evaluation of the model with r2
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2_score(y_test, tahmin)
# 0.7878388273269488
plt.scatter(y_test, tahmin, color='red')

# pca dönüşümünden sonra gelen SVR
from sklearn.svm import SVR
svr_reg2 = SVR()
svr_reg2.fit(x_train_2, y_train)
tahmin2 = svr_reg2.predict(x_test_2)

#evaluation of the model with r2
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2_score(y_test, tahmin2)
#  0.7957258534317097
plt.scatter(y_test, tahmin2, color='red')




