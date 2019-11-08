import os
import pandas as pd
# check the current working directory
os.getcwd()
# change working directory
os.chdir("C:\\Users\\pc\\Desktop\\data analyst classes")
# Loading the data
dataset = pd.read_csv("logisticmodel.csv")
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
# define Rank variable as a categorical variable and consider rank 4 as reference category
#Reference Category
from patsy import dmatrices, Treatment
y, X = dmatrices('admit ~ gre + gpa + C(rank, Treatment(reference=4))', dataset, return_type = 'dataframe')
y.shape
X.shape
X.head()
# Split Data into two parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Build Logistic Regression Model
#Fit Logit model
import statsmodels.api as sm
logit = sm.Logit(y_train, X_train)
result = logit.fit()

#Summary of Logistic regression model
result.summary()
result.params
# Confusion Matrix and Odd Ratio
#Confusion Matrix
cnf_matrix =result.pred_table()
cnf_matrix
#Odd Ratio
import numpy as np
np.exp(result.params)
# Prediction on Test Data
#prediction on test data
y_pred = result.predict(X_test)
from sklearn import metrics

# import the metrics class
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#Calculate Area under Curve (ROC)
# AUC on test data
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc(false_positive_rate, true_positive_rate)

# visualize the confusion matrix
#import required modules
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
#Text(0.5,257.44,'Predicted label')

# # logistic model using sklearn 
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred1=logreg.predict(X_test)
# accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
print("Precision:",metrics.precision_score(y_test, y_pred1))
print("Recall:",metrics.recall_score(y_test, y_pred1))

f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)

#claa=ssification table
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
Text(0.5,257.44,'Predicted label')
# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
# gini
gini =2*auc-1
gini
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# KS
Test_Data1 = pd.concat([y_test,y_pred],axis =1)
Test_Data1.columns =["Dep_flag","Prob"]
Test_Data1.columns
Test_Data1['decile'] = pd.qcut(Test_Data1['Prob'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
Test_Data1.head()

Test_Data1.columns = ['Event','Probability','Decile']
Test_Data1.head()

Test_Data1['NonEvent'] = 1-Test_Data1['Event']
Test_Data1.head()


df1 =pd.pivot_table(data=Test_Data1,index=['Decile'],values=['Event','NonEvent','Probability'],aggfunc={'Event':[np.sum],'NonEvent': [np.sum],'Probability' : [np.min,np.max]})
df1.head()
df1.reset_index()

df1.columns = ['Event_Count','NonEvent_Count','max_score','min_score']
df1['Total_Cust'] = df1['Event_Count']+df1['NonEvent_Count']
df1
#  Sort the min_score in descending order.

df2 = df1.sort_values(by='min_score',ascending=False)
df2

df2['Event_Rate'] = (df2['Event_Count'] / df2['Total_Cust']).apply('{0:.2%}'.format)
default_sum = df2['Event_Count'].sum()
nonEvent_sum = df2['NonEvent_Count'].sum()
df2['Event %'] = (df2['Event_Count']/default_sum).apply('{0:.2%}'.format)
df2['Non_Event %'] = (df2['NonEvent_Count']/nonEvent_sum).apply('{0:.2%}'.format)
df2


df2['ks_stats'] = np.round(((df2['Event_Count'] / df2['Event_Count'].sum()).cumsum() -(df2['NonEvent_Count'] / df2['NonEvent_Count'].sum()).cumsum()), 4) * 100
df2

flag = lambda x: '*****' if x == df2['ks_stats'].max() else ''
df2['max_ks'] = df2['ks_stats'].apply(flag)
df2
df2.to_csv("ks_test.csv")

# Gains Chart
df_test1 = df2.copy()
df_test1['Event_cum%'] = np.round(((df_test1['Event_Count'] / df_test1['Event_Count'].sum()).cumsum()), 4) * 100
df_test1


df_test2 = df_test1[['Event_cum%']]
df_test2.reset_index()
df_test2.columns = ['Event_cum%_test']
df_test2

#df_train = df3[['Event_cum%']]
#df_train.reset_index()
#df_train.columns = ['Event_cum%_train']
#df_train2 = df_train.copy()
df_test2['Base %'] = [10,20,30,40,50,60,70,80,90,100]
df_test2

gains_chart = df_test2.plot(kind='line',use_index=False)
gains_chart.set_ylabel("Proportion of Event",fontsize=12)
gains_chart.set_xlabel("Decile",fontsize=12)
gains_chart.set_title("Gains Chart")

# Lift Chart

final2 = df_test2.copy()
final2['lift_test'] = (df_test2['Event_cum%_test']/df_test2['Base %'])
final2['Baseline']  = [1,1,1,1,1,1,1,1,1,1]
final2


lift_chart = final2[['lift_test','Baseline']]
lift_chart

lift_chart1 = lift_chart.plot(kind='line',use_index=False)
lift_chart1.set_ylabel("lift",fontsize=12)
lift_chart1.set_xlabel("Decile",fontsize=12)
lift_chart1.set_title("Lift Chart")
lift_chart1.set_ylim(0.0,2)

# 


