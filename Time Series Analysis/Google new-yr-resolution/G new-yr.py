# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:39:33 2018

@author: Rohit
"""
#Import all necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set()

#Use the skiprows arguement to remove redundant information
data=pd.read_csv("multiTimeline.csv",skiprows=1)

#Rename the columns
columns=['Month','Diet','Gym','Finance']
data.columns=columns

#Convert the 'Month' column to datetime attribute
data['Month']=pd.to_datetime(data['Month'])

#Reindex the columns so that it becomes useful for data visuaizations
data=data.set_index('Month',inplace=False)
print (data.info())

# Check for seasonality and trends in the data
fig0=data.plot()
plt.xlabel('Year')
plt.title('Data')
plt.show()

#Get the plots for all attributes and check for characteristics
fig1=data['Diet'].plot(color='blue',alpha=0.6)
plt.xlabel('Year')
plt.title('Diet')
plt.show()
fig2=plt.plot(data['Gym'],color='green',alpha=0.6)
plt.xlabel('Year')
plt.title('Gym')
plt.show()
fig3=data['Finance'].plot(color='red',alpha=0.6)
plt.xlabel('Year')
plt.title('Finance')
plt.show()

#We see that both Gym and Diet peak at the start of every year
#Further, Gym also has a clear trend 
#To trace the trend clearly, we use the method of rolling average which
#is one of the many ways to trace a trend in a data and plot it
data[['Diet']].rolling(12).mean().plot()
data[['Gym']].rolling(12).mean().plot()
data[['Finance']].rolling(12).mean().plot()
plt.show()

#Now we trace the seasonilty of data using the method of difference
data[['Diet']].diff().plot()
plt.show()

#Get the correlation matrix to find autocorrelation and plot it
corrmat=data.corr()
print (corrmat)

data.diff().plot()
print(data.diff().corr())

pd.plotting.autocorrelation_plot(data['Diet'])