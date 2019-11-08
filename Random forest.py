# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:49:25 2017

@author: kapil
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import tree
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
# check the current working directory
os.getcwd()
# change working directory
os.chdir("C://Users//visu//Desktop//R")

# Loading the data
dataset = pd.read_csv("german_credit.csv")
# explore data 
dataset.describe()
# dimension of data 
dataset.shape
# Number of rows
dataset.shape[0]
# number of columns
dataset.shape[1]
# name of columns
list(dataset)
# data detail
dataset.info()
#split dataset in features and target variable
x = dataset.iloc[:,1:21]
x.shape
y =dataset.iloc[:,0]
y
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
X_train.shape
X_test.shape
# Initializing and Fitting Model
rfc = RandomForestClassifier(n_estimators=30)  #n estimator means taking 30 tree
rfc_fit =rfc.fit(X_train,y_train)
# predition on train
rfc_pred_train =rfc_fit.predict(X_train)
# Accuracy on train
train_accuracy =metrics.accuracy_score(y_train,rfc_pred_train)
train_accuracy
# prediction on test
rfc_pred = rfc_fit.predict(X_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred)
test_accuracy
# Tuning Hyperparameters
# Grid Search
# Defining Parameters

params_RF = {"max_depth": [3,5,6,7,8], "max_features":['auto', 'sqrt', 'log2'],
"min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10],
"criterion": ["gini", "entropy"]}
params_RF
# Initializing, Building and Fitting Model
model_RF = GridSearchCV(RandomForestClassifier(), param_grid=params_RF)
model_RF.fit(X_train,y_train)
# Best Parameters
	
model_RF.best_params_
# Predict and Check Accuracy for train
rfc_pred_train1 =model_RF.predict(X_train)
train_accuracy1 =metrics.accuracy_score(y_train,rfc_pred_train1)
train_accuracy
# prediction on test
rfc_pred1 = model_RF.predict(X_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred1)
test_accuracy

# Random Search
# Parameters
params_RF_RS = {"max_depth": randint(3,8),
"max_features":['auto', 'sqrt', 'log2'], "min_samples_split":randint (2,10),
"min_samples_leaf":randint (1,10),
"criterion": ["gini", "entropy"]}
# Building and Fitting Model
RF_RS = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params_RF_RS,n_iter=100)
RF_RS.fit(X_train,y_train)
# Best Parameters
RF_RS.best_params_
# Predict and Check Accuracy
# Predict and Check Accuracy for train
rfc_pred_train2 =RF_RS.predict(X_train)
train_accuracy2 =metrics.accuracy_score(y_train,rfc_pred_train2)
train_accuracy
# prediction on test
rfc_pred2 = RF_RS.predict(X_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred2)
test_accuracy
#ks, gini, lift and gain