# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
	
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint
from scipy.stats import uniform
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
# Initializing and Fitting Model
NB = GaussianNB()
NB.fit(X_train,y_train)
#Predict and Check Accuracy

pred_NB = NB.predict(X_test)
metrics.accuracy_score(y_test,pred_NB)

#KS, lift, Gain,