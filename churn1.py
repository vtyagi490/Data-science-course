# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:56:22 2019

@author: Deepak Gupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.getcwd()
os.chdir("C:\\Users\\pc\\Desktop\\data analyst classes")
dataset=pd.read_csv("churn-data.csv")
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
dataset["churn"]=dataset["churn"].astype(int)
dataset["international plan"]=np.where(dataset["international plan"]=="yes",0,1)
dataset["voice mail plan"]=np.where(dataset["voice mail plan"]=="yes",0,1)
dataset=dataset.drop("phone number",axis=1)
dataset

y=dataset.iloc[:,19]
x=dataset.iloc[:,1:18]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=10)
logit=sm.Logit(y_train,x_train)
result=logit.fit()

result.summary()
result.params
cnf_matrix =result.pred_table()
cnf_matrix
import numpy as np
np.exp(result.params)
y_pred = result.predict(x_test)
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc(false_positive_rate, true_positive_rate)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred1=logreg.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
print("Precision:",metrics.precision_score(y_test, y_pred1))
print("Recall:",metrics.recall_score(y_test, y_pred1))
Accuracy=metrics.accuracy_score(y_test, y_pred1)
Precision=metrics.precision_score(y_test, y_pred1)
Recall=metrics.recall_score(y_test, y_pred1)

f1_score = 2*((Recall+Precision)/(Recall*Precision))

cnf_matrix1 = metrics.confusion_matrix(y_test, y_pred1)
cnf_matrix1
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix1), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
gini =2*auc-1
gini
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

test_data=pd.concat([y_test,y_pred],axis=1)
test_data.columns=["dep_flag","prob"]
test_data.columns
test_data['decile'] = pd.qcut(test_data['prob'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
test_data.head()
test_data.columns = ['Event','Probability','Decile']
test_data.head()
test_data['NonEvent'] = 1-test_data['Event']
test_data.head()
df1=pd.pivot_table(data=test_data,index=["Decile"],values=['Event','NonEvent','Probability'],aggfunc={'Event':[np.sum],'NonEvent': [np.sum],'Probability' : [np.min,np.max]})
df1.head()
df1.reset_index()
df1.columns= ['Event_Count','NonEvent_Count','max_score','min_score']
df1['Total_Cust'] = df1['Event_Count']+df1['NonEvent_Count']
df1
df2 = df1.sort_values(by='min_score',ascending=False)
df2



