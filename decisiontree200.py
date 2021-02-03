# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:56:39 2021

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


from sklearn.tree import DecisionTreeClassifier
#binary inducated
#minimum 2 instances in leaves
#minimum splitting of
#subsets up to 2 maximum tree depth of 200
dct = DecisionTreeClassifier(criterion='entropy', max_depth=(200),min_samples_split=2,min_samples_leaf=2)

dct.fit(X_train,y_train)

y_pred = dct.predict(X_test)





from sklearn.model_selection import cross_val_score
#model = classifier
X = X_train
y = y_train
print(cross_val_score(dct,X,y,cv=10))
scores = absolute(cross_val_score(dct,X,y,cv=10))
#ortalama mutlak hata
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

#from sklearn.metrics import accuracy_score
#acc_dct=accuracy_score(y_test,y_pred)
#print('dct accuracy')
#print(acc_dct)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# Accuracy
from sklearn.metrics import accuracy_score
acc_dct=accuracy_score(y_test, y_pred)
print('dct accuracy')
print(acc_dct)
# Recall
from sklearn.metrics import recall_score
recall_dct=recall_score(y_test, y_pred, average=None)
print('dct recall')
print(recall_dct)
# Precision
from sklearn.metrics import precision_score
precision_dct=precision_score(y_test, y_pred, average=None)
print('dct precision')
print(precision_dct)

from sklearn.metrics import f1_score
F1=f1_score(y_test, y_pred, average=None)
print('f1 score')
print(F1)
#m= confusion_matrix(y_test, y_pred)
#rint(cm)