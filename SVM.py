# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:36:57 2019

@author: pc
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import randint
from scipy.stats import uniform

#check the current working directory
os.getcwd()
#change directory
os.chdir('C:\\Users\\pc\\Desktop\\data analyst classes')
#Loading data
dataset = pd.read_csv("german_credit.csv")
#explore data
dataset.describe()
#dimenssion
dataset.shape
#data details
dataset.info()
# spliting the data in feature and target variable
x = dataset.iloc[:, 1:2]
x.shape
y = dataset.iloc[:, 0]
y.shape
# spliting dataset into training and test
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.3, random_state= 1)
X_train.shape
Y_test.shape
#intializing and fiting the model
model_SVM = SVC()
model_SVM.fit(X_train, Y_train)

#Predict  and check accuracy
pred_svm = model_SVM.predict(X_test)
metrics.accuracy_score(Y_test, pred_svm)

#Turning Hyperparameters
# Grid Search
params_SVM = {"C": [0.01,0.1, 1], 'kernel' : ['linear','rbf']}
model_svm_GS = GridSearchCV(SVC(), param_grid= params_SVM)
model_svm_GS.fit(X_train, Y_train)
# Best Parametres
model_svm_GS.best_params_
#Predict and check accuracy
pred_svm_GS= model_svm_GS.predict(X_test)
metrics.accuracy_score(Y_test, pred_svm_GS)

#Random Search
params_RS1 = {'C': uniform(0.01, 1), 'kernel' : ['linear', 'rbf']}
# BUilding  and finting model
SVM_RS = RandomizedSearchCV(SVC(), param_distributions= params_RS1, n_iter =100)
SVM_RS.fit(X_train, Y_train)

#Best parameters
SVM_RS.best_params_
# Preidict and check accuracy
pred_svm_rs = SVM_RS.predict(X_test)
metrics.accuracy_score(Y_test, pred_svm_rs)
