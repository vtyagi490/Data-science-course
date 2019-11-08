# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:53:14 2019

@author: pc
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
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# check the current working directory
os.getcwd()
# change working directory
os.chdir("C:\\Users\\pc\\Desktop\\data analyst classes")

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


#intializing and fitting MOdel
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#Predict and check accuracy
pred_knn = knn.predict(X_test)
metrics.accuracy_score(y_test, pred_knn)

#Tuning Hyperparameters
#Grid search

#Defining parameters
params_knn = {"n_neighbors": [5,6,7,8,9,10,12], 'leaf_size':[1,2,3,5], 'weights':['uniform', 'distance'],
              'algorithm':['auto','ball_tree','kd_tree','brute']}

#initializing , building and fitting MOdel
model_knn_GS = GridSearchCV(KNeighborsClassifier(), param_grid=params_knn)
model_knn_GS.fit(X_train, y_train)

#Best Parameters
model_knn_GS.best_params_

#Predict and check Accuracy
pred_knn_GS = model_knn_GS.predict(X_test)
metrics.accuracy_score(y_test, pred_knn_GS)

#Random Search

params_knn_rs = {'n_neighbors': sp_randint(5,12),
                 'leaf_size': sp_randint(1,5),'weights': ['uniform', 'distance'],
                 'algorithm': [ 'auto', 'ball_tree', 'kd_tree', 'brute']}

#Building and fiting model
KNN_RS = RandomizedSearchCV(KNeighborsClassifier(),
                            param_distributions= params_knn_rs, n_iter= 250)
KNN_RS.fit(X_train, y_train)

#Best Parameters
KNN_RS.best_params_
#Predict and check Accuracy
pred_knn_rs = KNN_RS.predict(X_test)
metrics.accuracy_score(y_test, pred_knn_rs)
