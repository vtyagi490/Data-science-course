# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:52:15 2019

@author: VISHAL
"""

#1D Array
import numpy as np
a=np.array([15,25,14,78,96])
a
type(a)
 # changing the datatype
a.dtype
a=np.array([15,25,14,78,96],dtype = "float")
a
a.dtype
# Creating the sequence of numbers
b=np.arange(start = 20, stop= 30)
b
# Create an Arthmetic Progrression
c=np.arange(20,30,3)   # 30 is excluded
c
# Reshaping the arrays
f= np.arange(101,113)
f
f.reshape(3,4)
f
# To modify the shape of array
f.resize(3,4)
f
# Missing Data
import numpy as np
val= np.array([15,10,np.nan,3,2,5,6,4])
val
val.sum()
# TO Ignore ,missing data
np.nansum(val)
np.isnan(val)
# 2D arrays
g= np.array([(10,20,30),(40,50,60)])
#Alternatively
g=np.array([[10,20,30],[40,50,60]])
g
 
g.ndim  # tells no. of rows
g.size  
g.shape
 # Creating some usual matrices
np.zeros( (2,4))
a=np.zeros([2,4],dtype=np.int16)
a
 #  To get a matrix of all random numbers from 0 to 1
a=np.empty((3,4))
a
# to create a matrix of unity 
np.ones([3,3])
# to create a diagonal matrix
np.diag([114,15,16,17])
# Reshaping 2D arrays
g=np.array([(10,20,30),(40,50,60)])
# To get a flattened 1D array
import numpy as np
g.ravel()
# returns the array with a modified shape 
g.reshape(3,-5)
g.shape
# Resize ( ) will modify the shape in the original array
g.resize( (3,2))
g
# create some arrays A,b an B
A=np.array([[2,0,1],[4,3,8],[7,6,9]])
b=np.array([1,101,14])
B=np.array([[10,20,30],[40,50,60],[70,80,90]])
# Get the transpose, trace, and inverse
A
A.T
# Transpose
A.transpose()  # or A.T

## Trace
np.trace(A)  # add the diagonal number

#Inverse 
np.linalg.inv(A)
# Matrix addition and subtraction 
A+B
A-B
# Matrix Multiplication
A.dot(B)
A
# square
np.absolute(B)
np.sqrt(B)
np.exp(B)

A=np.arange(1,10).reshape(3,3)
A
A.sum()
A.min()
A.max()
A.mean()
A.std()
A.var()
# To   obtain the index of the minimuman and max elements
A.argmin()
A.argmax()
## find the above statistics for each row or column
A.sum(axis=0)
A.mean()
A.std(axis=0)
A.argmin(axis=0)
A.min(axis=1)
A.argmax(axis=0)
# find the cumulative sum along each row
A.cumsum(axis=1)
# Indexing in  arrays
x = np.arange(10)
x
x[2]
x[2:5]
# find out position of elements that satisfy a given condition
a=np.array([8,3,7,0,4,2,5,2])

np.where(a>4)
# Indexing with Array of indices
x=np.arange(11,35,2)
x
# Array {Which subset the elements of x}
i=np.array( [0,1,5,3,7,9])
x[i]
# Create a 2D array j of indices to subset x.
j = np.array( [ [0,1], [6,2]])
x[j]
# indexing with Boolean arrays
a = np.arange(12).reshape(3,4)
a
b=a>4
b
# Note That 'b' is a Boolean with same shape as that of 'a'.
# a select the elements from 'a' which adhere to condition 'b'
a[b]
a[b]=0
a
