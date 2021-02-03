# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:34:06 2021

@author: alibs
"""

from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from numpy import absolute

dataFrame = pd.read_excel("fruitdataset.xlsx")

y = dataFrame[["apple","cherries","mango","orange","pineapple"]].values
X = dataFrame[["size","width","height"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=40)
#print(X_train.shape)
#ölçeklendir
scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=0.1)


model.fit(X_train, y_train)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


# ACCURACY
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))