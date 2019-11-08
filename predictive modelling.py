# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:13:17 2019

@author: Deepak Gupta
"""

import numpy as np
a=np.array([1,2,3,4,5,6,7,9,53,478,908])
p=np.percentile(a,25)
p
un1=[8,9,10,8,7,7,8]
un2=[14,2,8,6,13,4,10]
q1=np.percentile(un1,25)
q3=np.percentile(un1,75)
iqr=q3-q1
iqr
q1=np.percentile(un2,25)
q3=np.percentile(un2,75)
iqr2=q3-q1
iqr2
lower_boundry=(q1-(1.5)*iqr)
upper_boundry=q3+1.5*iqr

#calculated mad(mean abs deviation) of series a calcuted wich one is batter(low one is batter)
un1=[8,9,10,8,7,7,8]
un2=[14,2,8,6,13,4,10]
def my_fun(un1,un2):
    a=np.mean(un1)
    b=np.mean(un2)
    mad1=((un1-a)/len(un1))
    mad1=np.abs(mad1)
    d=np.sum(mad1)
    mad2=((un2-b)/len(un2))
    mad2=np.absolute(mad2)
    f=np.sum(mad2)
    if d>f:
        print(f)
    else:
        print(d)
        
my_fun(un1,un2)
  #****************************************  
 #alculate variance(low one is batter)
 t=[8,9,10,11,1,13]
 t1=[9,10,65,89,24]
np.var(t)
np.std(t)
np.var(t1)
np.std(t1)
#********************************
#measure of shape
from scipy.stats import kurtosis ,skew
#creat randon variable baesd  on a normal distribution
x=np.random.normal(0,2,10000)
x
a=kurtosis(x)
print(a)
b=skew(x)
print(b)
#******************************8
#hypothesis testing
#t test  one veriable
#simple t test
import pandas as pd
from scipy import stats
mydata=pd.read_excel("E:\\hypothesis.xlsx")
mydata
mydata.columns

result=stats.ttest_1samp(mydata.loc[:,"lifetime_yrs"],10)
print(result)

#*************************************************
#indepandent t test
mydata=pd.read_excel("E:\\example2.xlsx")
list(mydata.columns)
result1=stats.ttest_ind(mydata.loc[:,"city"],mydata.loc[:,"rating"])
print(result1)
#************************************************
#paired t test
mydata1=pd.read_excel("E:\\example3.xlsx")
result1=stats.ttest_rel(mydata1.loc[:,"before"],mydata1.loc[:,"after"])
print(result1)
#***************************************************8
#indepandent t test
mydata3=pd.read_excel("E:\\example4.xlsx")
result=stats.ttest_ind(mydata3.loc[:,"lightsprouts"],mydata3.loc[:,"darksprouts"])
print(result)

#********************************************************8
#paired t test
mydata4=pd.read_excel("E:\\example5.xlsx")
result=stats.ttest_rel(mydata4.loc[:,"before"],mydata4.loc[:,"after"])
print(result)
#**********************************
#anova(one way anova)
import pandas as pd
import numpy as np
mydata=pd.read_excel("E:\\example6.xlsx")
import statsmodels.api as sm
from statsmodels.formula.api  import ols
mod=ols("sales~place",data=mydata).fit()
aov_tables=sm.stats.anova_lm(mod,typ=2)
print(aov_tables)
#post hoc anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc=MultiComparison(mydata["sales"],mydata["place"])
mc_results=mc.tukeyhsd()
print(mc_results)
#two way anova...............................
data=pd.read_excel("E:\\two way anova.xlsx")
formula="sales ~ QF + avail + QF:avail"
model=ols(formula,data).fit()
aov_table=sm.stats.anova_lm(model,typ=2)
print(aov_table)

#post hoc anova
mc1=MultiComparison(data["sales"],data["QF"])
mc2=MultiComparison(data["sales"],data["avail"])
mc1_results=mc1.tukeyhsd()
mc2_results=mc2.tukeyhsd()
print(mc1_results)
print(mc2_results)
#***************************************
#chi square test
data1=pd.read_excel("E:\\chi square test.xlsx")
chi=stats.chi2(data1["education"],data1["grades"])
print(chi)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#check the working directory
os.getcwd()
os.chdir("E:/")
#loding the data
dataset=pd.read_csv("mydata.csv")
#explore data
dataset.describe()
#dimension of data
dataset.shape
#number of rows 
dataset.shape[0]
#no of cols
dataset.shape[1]
#name of columns
list(dataset)
#data details
dataset.info()
#correlation
corr=dataset.corr()
print(corr)

fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,len(dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns)
ax.set_yticklabels(dataset.columns)
plt.show()
#div into indepandat(x) and depedant variable(y)
x=dataset.iloc[:,1:11]
x.shape
y=dataset.iloc[:,0]
y
#splitting the data into traning and test  sets
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=100)
#linear regression using sklearn
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm=lm.fit(x_train,y_train)
#coefficents

print(lm)
lm.coef_
#to satore cofficent in a data frame along  with thier  respectibe indepanted varivle
coefficents=pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))],axis=1)
#intecept
lm.intercept_
#to predict  the value of  y on the test set 
y_pred=lm.predict(x_test)
#residulas or error
y_error=y_test-y_pred
#r square
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
#calculate sum of square error
ss_residual=sum((y_test-y_pred)**2)
ss_total=sum((y_test-np.mean(y_test))**2)
r_squared=1-(float(ss_residual))/ss_total
r_squared
#add r square
adjusted_r_square=1-(1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
#calculate rmse
rmse=np.sqrt(np.mean(ss_residual))
rmse
#linear regression  using statsmodels
import statsmodels.api as sma
#lets add an intercept beta_0 to our model
X_train=sma.add_constant(x_train)
X_test=sma.add_constant(x_test)
#linear regression can be run by using ols
import statsmodels.formula.api as sm
lm2=sm.OLS(y_train,X_train).fit()
#summary
lm2.summary()
#predicted valures for test set
y_pred2=lm2.predict(X_test)
#detecting outlier
influence=lm2.get_influence()
resid_student=influence.resid_studentized_external
#combining train and residuals

resid=pd.concat([X_train,pd.Series(resid_student,name="Studentized residuals")],axis=1)
resid.head()
#absolute studentized residuals more than 3 is an outlires
resid.loc[np.absolute(resid["Studentized residuals"])>3,:]
ind=resid.loc[np.absolute(resid["Studentized residuals"])>3,:].index
ind
#dropping outlier
y_train.drop(ind,axis=0,inplace=True)
x_train.drop(ind,axis=0,inplace=True)
#intercpt columns is not thre
X_train.drop(ind,axis=0,inplace=True)
#intercept columns is thre
#dtecting  and removing multicollineraty #use statsmodel library to calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(x_train.values,j) for j in range(x_train.shape[1])]
#Vif is more than 5 for a particlaur variable then  that varible will be removed
def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)
    
    
train_out = calculate_vif(x_train)     
#Removing the variables from the test set
x_test.head()
x_test.drop(["cyl","hp","drat","wt","qsec","gear","carb"],axis = 1,inplace = True)
x_test.head()
# linear regression again on our new training set 
# let's add an intercept (beta_0) to our model
train_out = sma.add_constant(train_out)
X_test = sma.add_constant(x_test)
lm2 = sm.OLS(y_train,train_out).fit()
lm2.summary()
# Checking normality of residuals
# Shapiro Wilk test
from scipy import stats
stats.shapiro(lm2.resid)
# p value should be more than 0.05
# Checking for autocorrelation
# Ljungbox test
from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(lm2.resid , lags = 1) 
# p value sholud b emore themn 0.05
# Checking heteroscedasticity Using Goldfeld Quandt 
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(lm2.resid, lm2.model.exog)
lzip(name, test)
# p value should be more then 0.05
#logistic regression
























