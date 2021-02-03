# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:53:56 2020

@author: alibs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import DistanceMetric
import pickle
#2veri on ısleme
#2.1verilerin yuklenmesi
veriler = pd.read_csv('fruitdataset3.csv')
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
print(x)
y = veriler.iloc[:,0:1].values #bağımsız değişkenler
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

#veri olcekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#oznitelik olcekleme 

#farklı oznitelikleri aynı şekle sokma 
#ornegin boy metriği ve kilo metrigi farkılı
#aynı şekle sokacağız
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# 5 neighbours 
#mahalanobis distance metric 
#uniform weight distribution.
from sklearn.neighbors import KNeighborsClassifier
X  =  np . rastgele . rand ( 10 ; 2 )
cov  =  np . cov ( X , rowvar = False )

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform' ,metric='mahalanobis',metric_params=dict(V=cov))

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)





from sklearn.model_selection import cross_val_score
#model = classifier
X = X_train
y = y_train
print(cross_val_score(knn,X,y,cv=10))


cm= confusion_matrix(y_test, y_pred)
print(cm)