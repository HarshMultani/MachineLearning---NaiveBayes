# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 22:02:45 2020

@author: 138709
"""

# Naive Bayes Algorithm

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values


#Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)


# Predict the test set results
Y_pred = model.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)