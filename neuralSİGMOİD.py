# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:27:53 2021

@author: alibs
"""

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

dataFrame = pd.read_excel("fruitdataset.xlsx")

y = dataFrame[["apple","cherries","mango","orange","pineapple"]].values
X = dataFrame[["size","width","height"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=40)
print(X_train.shape)
#ölçeklendir
scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
#100 Neuron ve 2 Katman
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(100,activation="sigmoid"))

model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=10, batch_size=8, verbose=1)



# ACCURACY
score = model.evaluate(X_test, y_test,verbose=1)

print(score)