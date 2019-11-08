#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# check the current working directory
os.getcwd()
# change working directory
os.chdir("C:\\Users\\pc\\Desktop\\data analyst classes")
# Loading the data
dataset = pd.read_csv("mydata.csv")
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
# correlation 
corr = dataset.corr()
print(corr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns)
ax.set_yticklabels(dataset.columns)
plt.show()
# divided into independent (x) and dependent variables (y)
x= dataset.iloc[:,1:11]
x.shape
y =dataset.iloc[:,0]
y
 # Splitting the data into training and test sets
 import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test =train_test_split(x,y,test_size=.2, random_state =100)
# linear regression using sklearn
from sklearn.linear_model import LinearRegression
lm =LinearRegression()
lm= lm.fit(x_train,y_train)
# coefficients
lm.coef_
# To store coefficients in a data frame along with their respective independent variables
coefficients=pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)
print(coefficients)
# intercept
lm.intercept_
# To predict the values of y on the test set
y_pred = lm.predict(x_test)
# residuals or errow
y_error = y_test - y_pred
# R square
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
# calculate sum of square error
SS_Residual = sum((y_test-y_pred)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
r_squared
# adj r square
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
adjusted_r_squared
# calculate rmse 
rmse = np.sqrt(np.mean(SS_Residual))
rmse
#2. linear regression using statsmodels
 import statsmodels.api as sma
## let's add an intercept (beta_0) to our model
X_train = sma.add_constant(x_train) 
X_test = sma.add_constant(x_test) 
# Linear regression can be run by using sm.OLS:
import statsmodels.formula.api as sm  ## for python 3.7
import statsmodels.regression.linear_model as sm  ##for python 3.6
lm2 = sm.OLS(y_train,X_train).fit()
# summary 
lm2.summary()
# predicted values for test set
y_pred2 = lm2.predict(X_test) 
# Detecting Outliers: 
influence = lm2.get_influence()  
resid_student = influence.resid_studentized_external


# combining train and residuals

resid = pd.concat([x_train,pd.Series(resid_student,name = "Studentized Residuals")],axis = 1)
resid.head()
# absolute studentized residuals more than 3 is an outlier
resid.loc[np.absolute(resid["Studentized Residuals"]) > 3,:]
ind = resid.loc[np.absolute(resid["Studentized Residuals"]) > 3,:].index
ind
# Dropping Outlier 
y_train.drop(ind,axis = 0,inplace = True)
x_train.drop(ind,axis = 0,inplace = True)  #Interept column is not there
X_train.drop(ind,axis = 0,inplace = True)  #Intercept column is there
# Detecting and Removing Multicollinearity 
# use statsmodels library to calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])]
# VIF is more than 5 for a particular variable then that variable will be removed.
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

train_out.head()
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