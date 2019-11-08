# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:44:05 2019

@author: Deepak Gupta
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

os.getcwd()
os.chdir("E:/")
dataset=pd.read_csv("german_credit.csv")
dataset.describe()
dataset.shape
dataset.shape[0]
dataset.shape[1]
list(dataset)
dataset.info()
x=dataset.iloc[:,1:21]
x.shape
y=dataset.iloc[:,0]
y
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
X=X_train.columns
#visualizing Decison Trees
dot_data=tree.export_graphviz(clf,out_file=None,max_depth=3,feature_names=X,class_names=["1","0"],True,special_characters=True)
graph2=graphviz.Source(dot_data)
graph2.render("final")
 #grid search
params={"max_features":["auto","sqrt","log2"],"min_samples_split":[2,3,4,5,6,7,8,9,10],
        "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10,11],"max_depth":[2,3,4,5,6,7,8,9]}
params
DTC=tree.DecisionTreeClassifier()
DTC1=GridSearchCV(DIC,param_grid=params)
DTC1.fit(X_train,y_train)

