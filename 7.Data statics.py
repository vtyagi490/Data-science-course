# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:04:36 2019

@author: VISHAL
"""

import numpy as np
a=[2,3,5,4,5,6,8,20]
np.mean(a)
np.median(a)
a=np.array([1,2,3,4,5,6,7,8,9,53,478,908])
p=np.percentile(a,50)   # percentile
p
d=p/20  # second decile
d
d=np.quantile(0,1)
un1=[8,9,10,8,7,7,8]
un2=[14,2,8,6,13,4,10]
w=np.min(un2)
x=np.max(un2)
range=x-w
range
q3=np.percentile(un1,75)
q3
q1=np.percentile(un1,25)
q1
interquartile_range=q3-q1
interquartile_range
# calculation of outliers
a=[1,2,3,5,10,50]
q3=np.percentile(a,75)
q3
q1=np.percentile(a,25)
q1
iqr=q3-q1
iqr
#1.calculate lower boundary
lb=q1-1.5*iqr
lb
#2. calculate upper boundary
ub=q3+1.5*iqr
ub
# CAlULATIONS OF MAD(MEAN ABSOLUTE DEVIATION)
import numpy as np

T=[8,9,10,8,7,7,8]
  x=np.mean(T)
  MAD=(T-x)/len(T)
  MAD=np.absolute(MAD)
  MAD
  np.sum(MAD)
U=[14,2,8,6,13,4,10]
  x=np.mean(U)
  MAD=(U-x)/len(U)
  MAD=np.absolute(MAD)
  MAD
  np.sum(MAD)
#NOTE==HERE T is better becouse its value is less then U

#Calulation of variance
T=[8,9,10,8,7,7,8]
 np.var(T)  
U=[14,2,8,6,13,4,10]
 np.var(U)  #lesser value is good

#Calulation of standard deviation
U=[14,2,8,6,13,4,10]
 np.std(U)
T=[8,9,10,8,7,7,8]
 np.std(T)  #lesser std value is better
 
#Measure of Shape
 #1. by skewness
from scipy.stats import kurtosis, skew
 #create random values based on a normal distribution
x= np.random.normal(0,2,10000)
x 
a= kurtosis(x)
print(a)

#skewness
b=skew(x)
print(b)
np.mean(x)
np.median(x)
np.var(x)
np.std(x)

# one sample test
import pandas as pd
k=pd.read_csv('C:\\Users\\VISHAL\\Desktop\\data analyst classes\\lot.txt')
k
from scipy import stats
result=stats.ttest_1samp(k.loc[:,"lifetime_year"],10)
print(result)
