# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:08:34 2019

@author: VISHAL
"""

#import csv file
import pandas as pd
import numpy as np
mydata = pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data.csv")
mydata
mydata.columns
list(mydata)
print(mydata.head())
#No header in row  data file
mydata1 =pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data.csv",header =None)
mydata1.head()
# Add Column Names
mydata2 =pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data.csv",header =None, names = ['admit1','gre1','gpa1','rank1'])
mydata2.head()
mydata2.columns
# Specify missing values
mydata=pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data.csv",na_values=['.',"@"])
mydata.head()
#Skip first rows while importing csv file
mydata= pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data.csv",skiprows=5)
mydata

#
mydata=pd.read_excel("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data1.xlsx",nrows=10)
mydata
mydata=pd.read_excel("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\data1.xlsx",nrows=10,usecols=(0,3))
mydata
#IMPORT FILE FROM URL
mydata=pd.read_csv("http://winterolympicsmedals.com/medals.csv")
mydata
#READ EXCEL FILE
mydata=pd.read_excel("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\datal.xlsx")
mydata
#READ TEXT FILE
mydata=pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\agedata.txt")
mydata
#READ SAS FILE
mydata4=pd.read_sas("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\sas.sas7bdat")
mydata4.describe()
mydata4.info()
# CODING UTF
import pandas as pd
data=pd.read_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\sampledata.csv")
data.shape
data.shape[0]
# GET VARIABLE NAME
data.columns
list(data)
data.head()
#KNOWING THE VARIABLE TYPE
type(data)
data.dtypes
data['State'].dtypes
# CHANGING THE DATA TYPE
data.Y2008 = data.Y2008.astype(float)
data.dtypes
# HOW To SAVE DATA OUT SOURCE PATH
data.to_csv("C:\\Users\\VISHAL\\Desktop\\data analyst classes\\mydatatsave.csv")
# EXTRACT UNIQUE VALUE FROM DATA
data.Index.unique()
# FIND THE NO. OF UNIQUE VALUE
data.Index.nunique()
data.State.nunique()
# Gentrate Cross Tab
pd.crosstab(data.Index,data.State)
#frequency distribution
data.Index.value_counts(ascending = True)
# To Get Random Sample
dat1=data.sample(n=5)
dat1.shape
data.sample(frac = 0.1)
# select few columns
d1=data.loc[:,["Index","State","Y2008"]]
d1.shape
list(d1)
# selecting consecutive columns
data1=data.loc[:,"Index":"Y2008"]
list(data1)
dat6=data.iloc[:,0:5]
data2=data[["Index","State","Y2008"]]
data2.shape
# Renaming the variable
dataset = pd.DataFrame({"A":["John","Mary","Julia","Kenny","Henry"],"B":["Libra","Capricon","Aries","Scorpio","Aquarius"]})
dataset
# Renaming all the variables.
dataset.columns= ['Name','Zodiac Sign']
dataset
# Renaming only some of the variable
dataset.rename(columns = {"Name":"Cust_Name"},inplace = True)# we can use False but it will print only result , wont save it in memory
dataset
# REMOVING the columns and rows
d2 = data.drop('Index',axis =1) 
list(d2)
# Alternatively
data.drop("Index",axis = "columns")
d4=data.drop(['Index','State'],axis = 1)
list(d4)
d5 = data.drop(0,axis = 0)
data.shape
d5.shape
data.drop(50,axis='index')
d6=data.drop([0,1,2,3],axis =0)
d6.shape
# Sorting the data
dat1=data.sort_values("State",ascending=False)
print(dat1.head())
data.sort_values("State",ascending = False, inplace = True) # inplace is used when after some editing in data,if we want keep editing in that data
dat2 =data.Y2006.sort_values()
print(dat2.head())
 # Sort on 2 values'
dat3=data.sort_values(["Index","Y2002"])
dat3.head()
# Create New Variables
data["difference"]=data.Y2008-data.Y2009
list(data)
# Alternatively
data["differnce2"]=data.eval("Y2008-Y2009")
list(data)
data.head()
# creating data ratio
data["ratio"]=data.Y2008/data.Y2009
data.head()
list(data)
data1=data.assign(ratio = (data.Y2008/data.Y2009))
data1.head()
# Descriptive statistics
#for numeric variables
data.Y2009.describe()
data.describe()
#only for string / objects
data.describe(include =['object'])
data.Y2008.mean()
data.Y2008.median()
data.Y2008.min()
data.loc[:,["Y2002","Y2008"]].max()
# Groupby function
data.groupby("Index").Y2008.min()
data.groupby("Index")["Y2008","Y2010"].max()

a=data.groupby('Index').Y2002.agg(["count","min","max","mean","var"])
b=data.groupby("Index")["Y2002","Y2003"].agg(["count","min","max","mean"])
data.groupby("Index").agg({"Y2002":["min","max"],"Y2003":"mean"})
# Filtering
dat7=data[data.Index == "A"]
# Alternatively
d=data.loc[data.Index == "A",:]
data.loc[data.Index =="A","State"]
data.loc[data.Index == "A",:].State
# Filter the rows with index as "A" and income for 2002 > 1500000"
dat8=data.loc[(data.Index == "A") & (data.Y2002 > 1500000),:]

# filter the rows with index either "A" or "W"]
dat9 = data.loc[(data.Index == "A") | (data.Index == "W"),:]

#Alternatively
data.loc[data.Index.isin(["A","W"]),:]
data.query('Y2002>1700000 & Y2003 > 1500000')
data.query('Y2002>1700000 | Y2003>1500000')

# Dealing wiht missing values
import pandas as pd
import numpy as np
mydata = {'Crop': ['Rice','Wheat','Barley','Maize'], 'Yield' : [1010,1025.3,1404.2,1251.7], 'cost' : [102,np.nan,20,68]}
crops=pd.DataFrame(mydata)
crops
type(crops)
#SAME as is.na in R
crops.isnull() 
# opposite of previous command
crops.notnull()
# No. of missing values
crops.isnull().sum()
# Shows the rows with NAs.
crops[crops.cost.isnull()]
# Shows the rows with NAs in crops
crops[crops.cost.isnull()].Crop
# Shows the rows without NAs in crop
crops[crops.cost.notnull()].Crop

a=crops.dropna(how="any")

crops.dropna(how="all").shape

crops.dropna(subset = ['Yield',"cost"],how ='any').shape
crops.dropna(subset = ['Yield',"cost"],how = 'all').shape

#Replacing the missing values by "UNKNOWNS"
crops['cost'].fillna(value="UKNOWN",inplace=True)
crops
# Dealing with duplicates
data=pd.DataFrame({"Items":["TV","Washing Machine","Mobile","TV","TV","Washing Machine"],"Price":[10000,50000,20000,10000,10000,40000]})
data
data.loc[data.duplicated(),:]
data.loc[data.duplicated(keep="first"),:]
# Last enteries are not there, indices have changed.
data.loc[data.duplicated(keep="last"),:]
# all the duplicates, including unique are shown.

data.loc[data.duplicated(keep=False),:]
data.drop_duplicates(keep="first")
data.drop_duplicates(keep="last")
data.drop_duplicates(keep=False)

# if and else
import pandas as pd
students = pd.DataFrame({'Names':['John','Mary','Henry','Augustus','Kenny'],'Zodiac Signs':['Aquarius','Libra','Gemini','Pisces','Virgo']})

def name(row):
    if row["Names"] in ["John","Henry"]:
        return("yes")
    else:
        return("no")
        
    
#students.flag.describe()
flag5= students.apply(name,axis=1)
flag5
students
import numpy as np

students['flag']= np.where(students['Names'].isin(['John','Henry']),'yes','no')
students
#multiple conditions : if Else-if Else

def mname(row):
    if row["Names"] == "John" and row["Zodiac Signs"]== "Aquarius":
            
            return "yellow"
        
    elif row["Names"] == "Mary" and row["Zodiac Signs"]=="Libra":
              
            return "blue"
            
    elif row["Zodiac Signs"]== "Pisces" :
                
            return "red"
    else:
            return "black"
      
students['color']=students.apply(mname,axis=1)
students

## Merging or joining on the basis of common variable.

students = pd.DataFrame({'Names':['John','Mary','Henry','Maria'],'Zodiac Signs':['Aquarius','Libra','Gemini','Capricorn']})

students2=pd.DataFrame({'Names':['Vishal','Ketan','Mary','Henry','Kenny'],'Marks':[50,81,98,25,35]})

# intersecting result
result=pd.merge(students,students2, on='Names')
result
 # unions result
result1=pd.merge(students,students2,on='Names',how="outer")
result1
  #Left
result2=pd.merge(students,students2,on='Names',how="left")
result2
  #Right
result3=pd.merge(students,students2,on='Names',how='right',indicator=True)
result3
