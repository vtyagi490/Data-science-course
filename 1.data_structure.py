# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:00:36 2019

@author: VISHAL
"""
ll='hi how are you'
type(ll)
print(ll)
ll[2]
a=(ll.split()[1] )
print(a)
a=(ll.split('h'))
print(a)
x=[1,50,30,4,5,8,9,54,33,563,3555]
type(x)
x[11]
x[10]
x[-3]
k1 = [1,2,3]
k2 = [4,5,6]
z=k1+k2
print(z)

import numpy as np
z=np.add(k1,k2)
print(z)
# Repeat List N times
X=[1,2,3]
Z=X*3
print(Z)
# date 16.06.2019
#Add/Remove a list item
a = ['AA','BB','CC']
a
a.append('DD')
print(a)
a.remove('BB')
a
a.insert(0,"bb")
a
#sort list
k = [124,225,305,246,259]
k.sort() # it is ascending by default, here in () reverse=False is hidden
k
k.sort(reverse=True)
k
  #Tuple
K =(1,2,3,7,8)
type(K)
K[0]
State =("kk","ll","dd")
type(State) 
 # Tuple cannot be altered
X = (1,2,3)
X[2]=5
# Dictonary
teams = {'Dave' : 'team A','Tim' :'team B','Babita' : 'team C','Sam' : 'team B','Ravi':'team C' }
type(teams)
print(teams)
# Find value
teams["Babita"]
'Babita' in teams
#Delete an item
del teams['Ravi']
teams
# Add a team
teams['Deep'] = 'team B'
print(teams)
# Sets
##Sets are umordered collection of simple objects
X = set(['A','B','C'])
type(X)
# Does 'A' exits in set 
'A' in X
'D' in X



