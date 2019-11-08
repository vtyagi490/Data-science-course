# import libraries
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

# check the current working directory
os.getcwd()
# change working directory
os.chdir("E:/")

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
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
X=X_train.columns
#Visualizing Decision Trees
dot_data = tree.export_graphviz(clf, out_file=None, max_depth=3, feature_names=X, class_names=['1','0'],filled=True, rounded=True,special_characters=True)
graph2 = graphviz.Source(dot_data)
	
graph2.render("final")

# Grid Search

#Defining Parameters

params = {'max_features': ['auto', 'sqrt', 'log2'],'min_samples_split': [2,3,4,5,6,7,8,9,10], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],'max_depth':[2,3,4,5,6,7,8,9]}

params

# Initializing Decision Tree

DTC = tree.DecisionTreeClassifier()
# Building and Fitting Model

DTC1 = GridSearchCV(DTC, param_grid=params)
DTC1.fit(X_train,y_train)
# Best Parameters

modelF = DTC1.best_estimator_
modelF
# Predict and Check Accuracy
pred_modelF = modelF.predict(X_test)
metrics.accuracy_score(y_test,pred_modelF)
# Random Search
#Defining Parameters

from scipy.stats import randint as sp_randint

param_grid2 = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': sp_randint(2,10), 
          'min_samples_leaf': sp_randint(1,11),
         'max_depth':sp_randint(2,8)}
# Building and Fitting Model

DTC_RS = RandomizedSearchCV(DTC, param_distributions=param_grid2,n_iter=100)
DTC_RS1 = DTC_RS.fit(X_train,y_train)
DTC_RS1
# Best Parameters
	
DTC_RS1.best_params_
# Predict and Check Accuracy
pred_RS_DTC = DTC_RS1.predict(X_test)
metrics.accuracy_score(y_test,pred_RS_DTC)
#Calculate Area under Curve (ROC)
# AUC on test data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_RS_DTC)
auc(false_positive_rate, true_positive_rate)

# confusion matrix
ct =metrics.confusion_matrix(y_test,pred_RS_DTC)
ct
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#creat heatmap
sns.heatmap(pd.DataFrame(ct),annot=True,cmap"YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y=1.1)
plt.ylabel("actual label")
plt.xlabel("predicted label")
