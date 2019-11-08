# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:11:01 2019

@author: hp
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
# check the current working directory
os.getcwd()
# change working directory
os.chdir("F:\\R\\data")

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

GBC = GradientBoostingClassifier()
GBC.fit(X_train,y_train)
# Predict and Check Accuracy
pred_gradient = GBC.predict(X_test)
metrics.accuracy_score(y_test,pred_gradient)
# Tuning Hyperparameters
#Grid Search

#Defining Parameters

params_GB_GS = {"max_depth": [3,5,6,7],
              "max_features":['auto', 'sqrt', 'log2'], "min_samples_split": [2, 3, 10],
"min_samples_leaf": [1, 3, 10], 'learning_rate':[0.05,0.1,0.2],
'n_estimators': [10,30,50,70]}

# Initializing, Building and Fitting Model

model_Grad_GS = GridSearchCV(GradientBoostingClassifier(), param_grid=params_GB_GS)
model_Grad_GS.fit(X_train,y_train)

#Best Parameters
model_Grad_GS.best_params_
# Predict and Check Accuracy

pred_grad_GS = model_Grad_GS.predict(X_test)
metrics.accuracy_score(y_test,pred_grad_GS)
#Random Search
#Defining Parameters

params_GB_RS = {"max_depth":sp_randint(3,7),
              "max_features":['auto', 'sqrt', 'log2'], "min_samples_split": sp_randint(2,10), "min_samples_leaf": sp_randint(1,10), 'learning_rate':uniform(0.05,0.2),
'n_estimators':sp_randint(10,70)}

#Building and Fitting Model

Grad_RS = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=params_GB_RS,n_iter=100)
Grad_RS.fit(X_train,y_train)
#Best Parameters

Grad_RS.best_params_
#Predict and Check Accuracy

pred_Bag_RS = Grad_RS.predict(X_test)
metrics.accuracy_score(y_test,pred_Bag_RS)
