# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 20:16:21 2021

@author: alibs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from numpy import absolute
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

#random forest has 10 trees
#minimum split of subsets up to 5
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier( max_depth=(10), min_samples_split=5)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)


from sklearn.model_selection import cross_val_score
#model = classifier
X = X_train
y = y_train
print(cross_val_score(rfc,X,y,cv=10))
scores = absolute(cross_val_score(rfc,X,y,cv=10))
#ortalama mutlak hata
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

cm= confusion_matrix(y_test, y_pred)
# Accuracy
from sklearn.metrics import accuracy_score
acc_rfc=accuracy_score(y_test, y_pred)
print('rfc accuracy')
print(acc_rfc)
# Recall
from sklearn.metrics import recall_score
recall_rfc=recall_score(y_test, y_pred, average=None)
print('rfc recall')
print(recall_rfc[0])
# Precision
from sklearn.metrics import precision_score
precision_rfc=precision_score(y_test, y_pred, average=None)
print('rfc precision')
print(precision_rfc[0])

from sklearn.metrics import f1_score
F1=f1_score(y_test, y_pred, average=None)
print('f1 score')
print(F1[0])
