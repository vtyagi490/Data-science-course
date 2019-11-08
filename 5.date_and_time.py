# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:50:10 2019

@author: VISHAL
"""

# Get Current Date and Time
import datetime
date1 = datetime.date.today()
print(date1)

date = datetime.datetime.now()
print(date)

# Date object to represent a date
date3 = datetime.date(2019,3,13)
print(date3)

from datetime import date
date4= date(2019, 4, 13)
print(date4)
# Get current date
today = date.today()
today
print(today)
datetime.datetime.fromtimestamp(1300000)
# Get date from a timestamp
timestamp=date.fromtimestamp(1300000)
print('Date=',timestamp)
# Print today's year, month and day
today = date.today()
print(today)
print("Current year:",today.year)
print("Current month:",today.month)
print("Current day:",today.day)

# Time object to represent time
from datetime import time
time1 = time()
print(time1)

#
time2 = time(11,34,56)
print(time2)
time3 = time(hour =11, minute = 34, second =56)
print(time3)
time4 = time(11,34,56,234566)
print(time4)

#Print hour, minute, second and microsecond
time4= time(11,34,56)
print("hour =", time4.hour)
print("mintue=", time4.minute)
print("second=",time4.second)
print("microsecond=",time4.microsecond)
from datetime import datetime

# Differnce between two dates and times
from datetime import datetime, date

t1= date( year = 2018, month =7, day= 12)
t2 = date(year = 2017, month = 12, day=12)
t3= t1-t2
print(t3)
t4= datetime(year =2018, month=7, day=12, hour=7, minute =9, second=3)
t5=datetime(year = 2105, month= 6,day=12, hour=3, minute=22, second=7)
t6=t5-t4
print(t6)

type(t3)
type(t6)

#Differnce between two timedelta objects
from datetime import timedelta
 
t1=timedelta(weeks =2, days=5, hours=1, seconds=33)
t2=timedelta(days=4, hours=11, minutes=4, seconds=54)
t3=t1-t2 
t3
# Printing negative timedelta object
t1= timedelta(seconds=33)
t2= timedelta(seconds=54)
t3=t1-t2
print(t3)
print(abs(t3))

# Time duration in seconds
t= timedelta(days=5, hours=1, seconds=33, microseconds=233423)
print(t.total_seconds())
print(t.total_seconds()/3600)
#String to datetime
from datetime import datetime
date_string ="21 June, 2018"
print(date_string)
date_object = datetime.strptime(date_string,"%d %B, %Y") #if month is given in num or small form like Jun then we must use "b", and if year is given in two digit then we must use 'y'.

print(date_object)

# Handling timezone in Python
import pytz
local=datetime.now()
print(local.strftime("%m/%d/%Y, %H:%M:%S"))

tz_NY= pytz.timezone('America/New_York')
datetime_NY=datetime.now(tz_NY)
print(datetime_NY.strftime("%m/%d/%Y, %H:%M:%S"))

tz_London = pytz.timezone('Europe/London')
datetime_London = datetime.now(tz_London)
print(datetime_London.strftime("%m/%d/%Y, %H:%M:%S"))

from datetime import datetime

# Using datetime.strptime()
dt = datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")
dt
dt.strftime("%Y-%m-%d %H:%M")
dt.strftime("%y-%m-%d %H:%M")

dt.strftime("%A, %d.%B %Y %I:%M%p")
import time
import datetime
datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
datetime.datetime.now()
# Current year 
datetime.date.today().strftime("%Y")
#Month of year
datetime.date.today().strftime("%B")
# Week number of the year
datetime.date.today().strftime("%W")
#weekday of the week
datetime.date.today().strftime("%w")
#Day of the year
datetime.date.today().strftime("%d")
# Day of the week
datetime.date.today().strftime("%A")
# convert date time to regular format
date1=datetime.datetime.now()
reg_format_date = date1.strftime("%Y-%m-%d  %I:%M:%S%p")
reg_format_date
# Some other date formats
date5 = date1.strftime("%d %B %Y %I:%M:%S%p")
date5

date6 = date1.strftime("%Y-%m-%d  %H:%M:%S")
print(date6)