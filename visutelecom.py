# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:53:45 2019

@author: pc
"""

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

# Set Current directory
os.chdir("C:\\Users\\pc\\Desktop\\algo")
os.getcwd()

#import data
train = pd.read_csv("sample_data_intw.csv")
# explore data 
train.describe()
# dimension of data 
train.shape
# Number of rows
train.shape[0]
# number of columns
train.shape[1]
# name of columns
list(train)
# data detail
train.info()

################################     Exploratory Data Analysis ########################

# some variables in data are not useful, we will drop them As mobile number, pcircle is same, and Unnamed: 0 
train = train.drop(["msisdn","pcircle","Unnamed: 0"], axis=1)

# pdate variable is given as object converting it in requried format
train['pdate']= pd.to_datetime(train['pdate'])

## Derive separate fields like year, month, date,
train['day_of_month'] = train['pdate'].dt.day
train['month'] = train['pdate'].dt.month
# year is same so we didn't considered that.

#we have extracted relevant information from pdate variable remove this variable from the train dataset.
train = train.drop(['pdate'], axis=1)

###########################################  Missing Value Analysis    ###########################

train.isnull().sum().sum()    # No value is missing


################################# Outliers Analysis  ###################################################
desc = train.describe(include = 'all')
# Observation
#1. "aon, daily_decr30, daily_decr90,last_rech_date_da,last_rech_date_ma" variable have negative values which doesn't make any sense

for i in ['aon','daily_decr30', 'daily_decr90', 'last_rech_date_ma', 'last_rech_date_da']:
    print(i)
    print(Counter(train[i]<0))

# We are dropping these Negative values from variables
for i in ['aon','daily_decr30', 'daily_decr90', 'last_rech_date_ma', 'last_rech_date_da']:   
    train = train.drop(train[train[i]<0].index, axis=0)
    

# user defined function that will plot boxplot and distribution for four columns of dataset
from scipy.stats import norm

# As we have 35 variables in our train data set let's write a function which receives 4 variables at a time to display plots:

def histogram_and_box_plots(Var1, Var2, Var3, Var4, dataframe, bin1=50, bin2=50, bin3=50, bin4 =50, sup ="    "): 
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize= (16,8))
    super_title = fig.suptitle("Boxplot and Histogram: "+sup, fontsize=20)
    plt.tight_layout()
    sns.boxplot(y = Var1, data = dataframe, ax = ax[0][0])
    sns.boxplot(y = Var2, data = dataframe, ax = ax[0][1])
    sns.boxplot(y = Var3, data = dataframe, ax = ax[0][2])
    sns.boxplot(y = Var4, data = dataframe, ax = ax[0][3])
    sns.distplot(dataframe[Var1], kde=True, ax = ax[1][0], bins = bin1, fit=norm)
    sns.distplot(dataframe[Var2], kde=True, ax = ax[1][1], bins = bin2, fit=norm)
    sns.distplot(dataframe[Var3], kde=True, ax = ax[1][2], bins = bin3, fit=norm)
    sns.distplot(dataframe[Var4], kde=True, ax = ax[1][3], bins = bin4, fit=norm)
    fig.subplots_adjust(top = 0.9)
    plt.show()

# plotting boxplot and histogram for numerical variables in train dataset
histogram_and_box_plots('aon', 'daily_decr30', 'daily_decr90', 'rental30',dataframe = train, bin1 = 3,bin2 = 3,bin3 = 3,bin4 = 3,sup=" ")
histogram_and_box_plots('fr_ma_rech90', 'cnt_loans90', 'amnt_loans90','sumamnt_ma_rech90',dataframe = train, bin1 = 3,bin2 = 3,bin3 = 3,bin4 = 3,sup=" ")

# Get all column names
train.columns

cnamestrain = ['aon', 'daily_decr30', 'daily_decr90', 'rental30', 'rental90',
       'last_rech_date_ma', 'last_rech_date_da', 'last_rech_amt_ma',
       'cnt_ma_rech30', 'fr_ma_rech30', 'sumamnt_ma_rech30',
       'medianamnt_ma_rech30', 'medianmarechprebal30', 'cnt_ma_rech90',
       'fr_ma_rech90', 'sumamnt_ma_rech90', 'medianamnt_ma_rech90',
       'medianmarechprebal90', 'cnt_da_rech30', 'fr_da_rech30',
       'cnt_da_rech90', 'fr_da_rech90', 'cnt_loans30', 'amnt_loans30',
       'medianamnt_loans30', 'cnt_loans90', 'amnt_loans90',
       'medianamnt_loans90', 'payback30', 'payback90']

#2. maxamnt_loans30 and maxamnt_loans90 variables should have 12 and 6 values only, but some outliers are here
catnames = ['maxamnt_loans30','maxamnt_loans90']

for i in cnamestrain:   
# Quartiles and IQR
    q25,q75 = np.percentile(train[i],[25,75])
    IQR = q75-q25
# Lower and upper limits 
    LL = q25 - (1.5 * IQR)
    UL = q75 + (1.5 * IQR)
# Capping with ul for maxmimum values 
    train.loc[train[i] < LL,i] = LL 
    train.loc[train[i] > UL,i] = UL

################################ For 'maxamnt_loans30 and maxamnt_loans90' variable
# We are fiiling outliers of these variables with values(inlier = 6 & outliers = 12), according to our given limits
    
for j in catnames:
# Quartiles and IQR
    q25,q75 = np.percentile(train[j],[25,75])
    IQR = q75-q25
# Lower and upper limits 
    LL = q25 - (1.5 * IQR)
    UL = q75 + (1.5 * IQR)
# Capping with ul for maxmimum values 
    train.loc[train[j] < 6,j] = LL 
    train.loc[train[j] > 12,j] = UL

# Checking variables ' value range  
train['maxamnt_loans30'].unique()
train['maxamnt_loans90'].unique()
# Here we have now only two category 6 and 12.

desc = train.describe(include = 'all')
#3. Variables "'last_rech_date_da','cnt_da_rech30','fr_da_rech30','cnt_da_rech90','fr_da_rech90','medianamnt_loans30','medianamnt_loans90'"  have no informative datapoints so we will drop them.
 
train = train.drop(['last_rech_date_da','cnt_da_rech30','fr_da_rech30','cnt_da_rech90','fr_da_rech90','medianamnt_loans30','medianamnt_loans90'], axis=1)

#Creating dummies for each variable in 'maxamnt_loans90' and 'maxamnt_loans30' and merging dummies dataframe to train dataframe 
temp = pd.get_dummies(train['maxamnt_loans30'], prefix = 'maxamnt_loans30')
train = train.join(temp)
temp = pd.get_dummies(train['maxamnt_loans90'], prefix = 'maxamnt_loans90')
train = train.join(temp)

# As we creaate dummy according to formula (n-1), so we are dropping unnecessary variables.
train = train.drop(['maxamnt_loans30','maxamnt_loans90','maxamnt_loans30_12.0','maxamnt_loans90_12.0'], axis=1)
train.columns

################################### Feature selection  ##################################
#In this step we would allow only to pass relevant features to further steps. 
#We remove irrelevant features from the dataset. We do this by some statistical techniques,
# like we look for features which will not be helpful in predicting the target variables.
 
# Calculation of correlation between numerical variables
cnames =['aon', 'daily_decr30', 'daily_decr90', 'rental30', 'rental90',
       'last_rech_date_ma', 'last_rech_amt_ma', 'cnt_ma_rech30',
       'fr_ma_rech30', 'sumamnt_ma_rech30', 'medianamnt_ma_rech30',
       'medianmarechprebal30', 'cnt_ma_rech90', 'fr_ma_rech90',
       'sumamnt_ma_rech90', 'medianamnt_ma_rech90', 'medianmarechprebal90',
       'cnt_loans30', 'amnt_loans30','cnt_loans90','amnt_loans90', 'payback30', 'payback90']
 
df_num = train.loc[:,cnames]
corr = df_num.corr()
print(corr)

# plotiing the heatmap
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
plt.show()

# some independent variables are showing correlation together , let's drop them

#selecting upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
#find index of feature columns with correlation greater then 0.95
to_drop = [column for column in upper.columns if any (upper[column]>0.95)]
#drop marked feature
train.drop(train[to_drop], axis=1)

#Chisquare test of independence
#Save categorical variables
cat_names = ['maxamnt_loans30_6.0','maxamnt_loans90_6.0', "day_of_month"]
#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train['label'], train[i]))
    print(p)
# no value is greater then 0.05, no need to drop any variable
    

#####################################  Model Development  ##############################
# divided into independent (x) and dependent variables (y)
x= train.iloc[:,1:28]
x.shape
y =train.iloc[:,0]
y.shape

 # Splitting the data into training and test sets
x_train, x_test,y_train, y_test =train_test_split(x,y,test_size=.2, random_state =100)
print(train.shape, x_train.shape, x_test.shape,y_train.shape,y_test.shape)

#########################################  Logisctic Regression  ####################

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

#Model evaluation using confusion matrix
from sklearn import metrics
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
conf_matrix

from matplotlib import pyplot as plt
class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (10,10))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)
plt.ylabel('Actual label', fontsize =10)           
plt.xlabel('Predicted label', fontsize = 10)   

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))   #0.8857575092111363
print('Precision:', metrics.precision_score(y_test,y_pred))   #0.8895696427689115
print('Recall:', metrics.recall_score(y_test,y_pred))    #0.99414688017669

# F-1 score
Accuracy =  metrics.accuracy_score(y_test,y_pred)
Precision = metrics.precision_score(y_test,y_pred)
Recall = metrics.recall_score(y_test,y_pred)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)  #0.9389553834519805

#Area under Receiver Operating Curve(AUC)
y_pred_prob = logreg.predict_proba(x_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)   #0.8396978109722781
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))
plt.legend(loc = 4)
plt.show()

##########################################  Decision Tree  #####################
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred_clf = clf.predict(x_test)

conf_matrix_clf = metrics.confusion_matrix(y_test, y_pred_clf)
conf_matrix_clf

class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (8,8))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix_clf),
annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)
plt.ylabel('Actual label', fontsize =10)
plt.xlabel('Predicted label', fontsize = 10)

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,y_pred_clf))    #0.8849278969328747
print('Precision:', metrics.precision_score(y_test,y_pred_clf))  #0.9378474538581276
print('Recall:', metrics.recall_score(y_test,y_pred_clf))       #0.9315295416896742

# F-1 score
Accuracy =  metrics.accuracy_score(y_test,y_pred_clf)
Precision = metrics.precision_score(y_test,y_pred_clf)
Recall = metrics.recall_score(y_test,y_pred_clf)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)  #0.9346778214859548

from sklearn.metrics import auc

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_clf)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label='ROC area = %0.2f)' % roc_auc)
plt.legend(loc="lower right")
plt.show()

#Area under Receiver Operating Curve(AUC)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_clf)
auc = metrics.roc_auc_score(y_test, y_pred_clf)   #0.731038
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))
plt.legend(loc = 4)
plt.show()

#####################################  Random Forest  ###############################

from sklearn.ensemble import RandomForestClassifier
# Model building and fitting
rf = RandomForestClassifier()
rf = RandomForestClassifier(n_estimators =500).fit(x_train, y_train)
# Prediction on test data
y_pred_rf = rf.predict(x_test)

# Confusion Matrix
conf_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
conf_matrix_rf

# Confusion Matrix visualization
class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (8,8))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix_rf),annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)
plt.ylabel('Actual label', fontsize =10)
plt.xlabel('Predicted label', fontsize = 10)

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,y_pred_rf))   #0.9229924602884123 
print('Precision:', metrics.precision_score(y_test,y_pred_rf))   #0.934983949902647
print('Recall:', metrics.recall_score(y_test,y_pred_rf))    #0.9810877967973495

# F-1 Score
Accuracy=metrics.accuracy_score(y_test,y_pred_rf)   
Precision= metrics.precision_score(y_test,y_pred_rf)  
Recall= metrics.recall_score(y_test,y_pred_rf)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)  #0.9574812060463987

#Area under Receiver Operating Curve(AUC)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_rf)
auc = metrics.roc_auc_score(y_test, y_pred_rf)   #0.731038
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))
plt.legend(loc = 4)
plt.show()

# prediction on train
rfc_pred = rf.predict(x_train)
# train accuracy
train_accuracy = metrics.accuracy_score(y_train,rfc_pred)
train_accuracy   #0.9999633987884999  
# No overfitting problem


######KS Statics
# AS Random forest is perfoming well, now let's check KS value of this model.
# If Ks value lies in the range(40-70) and in top 3 decile, then its mean this model is working very well.

# Making Prediction on test data using random forest model
rfc_pred_proba= rf.predict_proba(x_test)
rfc_pred_proba=pd.DataFrame(rfc_pred_proba)
rfc_pred_proba1=rfc_pred_proba[1]

#taking y_test variable
y_test
y_test1=y_test.to_csv("y_test_label",index=False)
y_test1=pd.read_csv("y_test_label",header=None ,names=["a"])

# Merging y_test and probability dateset
Test_Data1 = pd.concat([y_test1,rfc_pred_proba1],axis =1)
Test_Data1.columns =["Dep_flag","Prob"]
Test_Data1.columns

# Createing Deciles in dataset
Test_Data1['decile'] = pd.qcut(Test_Data1['Prob'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
Test_Data1.head()

Test_Data1.columns = ['Event','Probability','Decile']
Test_Data1.head()

#Creating Non-event variable in dataset
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

# Calculating Ks statics value
df2['ks_stats'] = np.round(((df2['Event_Count'] / df2['Event_Count'].sum()).cumsum() -(df2['NonEvent_Count'] / df2['NonEvent_Count'].sum()).cumsum()), 4) * 100
df2
# Highlighting Decile in which our KS value lie Using lambda function
flag = lambda x: '*****' if x == df2['ks_stats'].max() else ''
df2['max_ks'] = df2['ks_stats'].apply(flag)
df2
df2.to_csv("ks_test.csv")

########################################  Conclusion  #########################################
#1. Random forest is performing best among all the above model. Becouse its f1_score, model accuracy is better then above all model
#2. Ks value is in top 3 decile. and lying within range 40 to 70.
#3. so we will choose Random forest for our buisness solution
