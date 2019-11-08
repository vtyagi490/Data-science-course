# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:38:16 2019

@author: VISHAL
"""

                        ###    loops   #####
#simple for loop'
nums = [1,2,3,4,5,6]
for i in nums:
    print(i)
    

# Prints out the numbers
for x in range(50):    
    print(x)
    
# Print out numbers of a range
    for x in range(3,10):
        print(x)
        
# Print out on an interval
        for x in range(30,80,2):
            print(x)
            
# While loops
            count =0
            while count < 10:
                print(count)
                count += 1
#################################################################                
# "Break" and " continue" statements
                
            count =0
            while True:
                print(count)
                count += 1
                if count >= 5:
                    break
            for x in range(10):
                # Check if x is even
                if x%2 == 0:
                    print(x)
                    continue
##########################################################
for i in range (1,10):
  if(i%5==0):
     print(i)
     continue
  else:
     print("this is not printed")
     
###########################################################     
#function to add two number
     def sum_two(a,b):
         return(a+b)
         
     sum_two(2,4)
#######################################################     
# return the absolute
     def abs_value(num):
         
         if num>=0:
             return num
         else:
             return -num
         
     abs_value(6)
     abs_value(-89)
#######################################################
# to get even numbers
list1=[10,21,4,45,66,93]
def even_num(x):
    for num in x:
        if num%2==0:
           print(num,"is even")
        else:
           print(num,"is odd")
                 
even_num(list1)
#######################################################
def my_function(country):
    print("I am from " + country)

my_function("India")  

##########################################################  

def my_function(x):
    return 10*x

my_function(10)
##########################################################
