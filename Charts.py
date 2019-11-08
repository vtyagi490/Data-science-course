# import library
import pandas as pd 
import matplotlib.pyplot as plt 
# x axis values
x = [1,2,3] 
# corresponding y axis values 
y = [2,4,1] 
# plotting the points  
plt.plot(x, y) 
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 

plt.title('My first graph!')
plt.show() 
 #line 1 points 
x1 = [1,2,3] 
y1 = [2,4,1] 
# plotting the line 1 points  
plt.plot(x1, y1, label = "line 1",color ='red') 
  
# line 2 points 
x2 = [1,2,3] 
y2 = [4,1,3] 
# plotting the line 2 points  
plt.plot(x2, y2, label = "line 2") 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 

# Customization of Plots
# x axis values 
x = [1,8,3,11,5,6] 
# corresponding y axis values 
y = [2,4,12,5,2,16] 
  
# plotting the points  
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
         marker='^', markerfacecolor='blue', markersize=10) 
  
# setting x and y axis range 
plt.ylim(1,18) 
plt.xlim(1,12) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Some cool customizations!') 
  
# function to show the plot 
plt.show() 

# crrate dataset
data = [['E001', 'M', 34, 123, 'Normal', 350], ['E002', 'F', 40, 114, 'Overweight', 450], ['E003', 'F', 37, 135, 'Obesity', 169], ['E004', 'M', 30, 139, 'Underweight', 189], ['E005', 'F', 44, 117, 'Underweight', 183], ['E006', 'M', 36, 121, 'Normal', 80], 
['E007', 'M', 32, 133, 'Obesity', 166],['E008', 'F', 26, 140, 'Normal', 120], 
['E009', 'M', 32, 133, 'Normal', 75], 
['E010', 'M', 36, 133, 'Underweight', 40] ] 
mydata =pd.DataFrame(data, columns = ['Emp_Id', 'Gender',  'Age', 'Sales', 'BMI', 'Income'])
mydata

# create histogram for numeric data 
mydata.hist()
# Show plot
plt.show()

# frequencies 
ages = [2,5,70,40,30,45,50,45,43,40,44, 
        60,7,13,57,18,90,77,32,21,20,40] 
  
# setting the ranges and no. of intervals 
range = (0, 100) 
bins = 10  
  
# plotting a histogram 
plt.hist(ages, bins, range, color = 'green', 
        histtype = 'bar', rwidth = 0.8) 
  
# x-axis label 
plt.xlabel('age') 
# frequency label 
plt.ylabel('No. of people') 
# plot title 
plt.title('My histogram') 
  
# function to show the plot 
plt.show() 
# bar plot
# Plot the bar chart for numeric values 
mydata.plot.bar()
plt.grid()
# plot between 2 attributes 
plt.bar(mydata['Age'],mydata['Sales'])
plt.xlabel("Age")
plt.ylabel("Sales") 
plt.grid()
plt.show()
# x-coordinates of left sides of bars  
left = [1, 2, 3, 4, 5] 
  
# heights of bars 
height = [10, 24, 36, 40, 5] 
  
# labels for bars 
tick_label = ['one', 'two', 'three', 'four', 'five'] 
  
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green']) 
  
# naming the x-axis 
plt.xlabel('x - axis') 
# naming the y-axis 
plt.ylabel('y - axis') 
# plot title 
plt.title('My bar chart!') 
  
# function to show the plot 
plt.show()  
#  Box plot chart
# For each numeric attribute of dataframe
mydata.plot.box() 
# # individual attribute box plot 
plt.boxplot(mydata['Income'],mydata["]) 
plt.boxplot(mydata['Sales']) 
plt.show() 
# Pie Chart 
plt.pie(mydata['Age'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},autopct =' %.1f %%', shadow = True) 
plt.show() 
plt.pie(mydata['Income'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}, autopct ='% 1.1f %%', shadow = True) 
plt.show() 

plt.pie(mydata['Sales'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}, 
autopct ='% 1.1f %%', shadow = True) 
plt.show() 

# defining labels 
activities = ['eat', 'sleep', 'work', 'play'] 
  
# portion covered by each label 
slices = [3, 7, 8, 6] 
  
# color for each label 
colors = ['r', 'y', 'g', 'b'] 
  
# plohelp(plt.pie)tting the pie chart 
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (.01, 1.011, 02.1, 1), 
        radius = 1.2, autopct = '%1.1f%%') 
  
# plotting legend 
plt.legend() 
  
# showing the plot 
plt.show() 
#  Scatter plot
# scatter plot between income and age 
plt.scatter(mydata['Income'], mydata['Age']) 
plt.show()
# scatter plot between income and sales 
plt.scatter(mydata['Income'], mydata['Sales']) 
plt.show() 
  
# scatter plot between sales and age 
plt.scatter(mydata['Sales'], mydata['Age'],label ="stars", color ="red", marker ="^", s=150) 
plt.show() 
  # x-axis values 
x = [1,2,3,4,5,6,7,8,9,10] 
# y-axis values 
y = [2,4,5,7,6,8,9,11,12,12] 
  
# plotting points as a scatter plot 
plt.scatter(x, y, label= "stars", color= "green",  
            marker= "*", s=30) 
  
# x-axis label 
plt.xlabel('x - axis') 
# frequency label 
plt.ylabel('y - axis') 
# plot title 
plt.title('My scatter plot!') 
# showing legend 
plt.legend() 
  
# function to show the plot 
plt.show() 
# Plotting curves of given equation
import numpy as np 
# setting the x - coordinates 
x = np.arange(0, 2*(np.pi), 0.1) 
# setting the corresponding y - coordinates 
y = np.sin(x) 
  
# potting the points 
plt.plot(x, y) 
  
# function to show the plot 
plt.show() 