# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:07:09 2018

@author: Rohit
"""

""" Iris dataset"""

# Classification of Iris dataset using logistic regression one vs. all method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Define all the parameters
alpha=0.2
epsilon=0.05

# Calculates the loss function
def geterror(x,y,theta):
    err=0
    for i in range(len(y)):
        hthetx=sigmoid(x[i],theta)
        t1=np.log(hthetx)
        t2=np.log(1-hthetx)
        err+=(y[i][0]*t1)+((1-y[i][0])*t2)
    err=-err/len(y)
    return err
    
# Define the sigmoid function
def sigmoid(x,theta):
    temp=x.dot(theta)
    result=math.exp(temp)
    denom=result+1
    return float(result/denom)

# Calculates the gradient of each parametrer
def getgradient(x,y,theta,idx):
    sum=0
    for i in range(len(x)):
        temp=sigmoid(x[i],theta)
        sum+=(temp-y[i][0])*x[i][idx]
    sum=sum/(len(x))
    return sum    

# Performs gradient descent and optimizes the loss function
def gradient_descent(x,y1,theta1,alpha):
    thetatemp1=theta1
    count=0
    while count<500:
        temp1=getgradient(x,y1,theta1,0)
        thetatemp1[0]=theta1[0]-alpha*temp1
        for i in range(4):
            temp=getgradient(x,y1,theta1,i+1)
            thetatemp1[i+1]=theta1[i+1]-alpha*temp
        theta1=thetatemp1
        if (abs(geterror(x,y1,theta1))<epsilon):
            break    
        count=count+1
    return theta1

# Prints the output which has the maximum probability
def predict(test,theta1,theta2,theta3):
    h1=sigmoid(test,theta1)
    h2=sigmoid(test,theta2)
    h3=sigmoid(test,theta3)
    maximum=max(list([h1,h2,h3]))
    output = "Nan"
    print (theta1,theta2,theta3)
    print (h1,h2,h3)
    if maximum==h1:
        output = 'Iris-setosa'
    elif maximum==h2:
         output = 'Iris-versicolor'
    elif maximum == h3:
        output = 'Iris-virginica'
    print (output)
    
# Load the data and rename the columns
data = pd.read_csv("iris.txt",sep=',',header=None,nrows=145)
columns = ["Sepal length","Sepal width","Petal length","Petal width","Class"]
data.columns=columns

# Segregate the data into input and output 
# There will be three output cases for three classes
inputdata = data[["Sepal length","Sepal width","Petal length","Petal width"]]
y1=data['Class'].replace(to_replace=['Iris-versicolor','Iris-virginica'],value=0)
y2=data['Class'].replace(to_replace=['Iris-setosa','Iris-virginica'],value=0)
y3=data['Class'].replace(to_replace=['Iris-versicolor','Iris-setosa'],value=0)

# Clean the data to aid the classification using logistic regression
# Replace the output columns with integers
y1=np.array(y1.replace(to_replace='Iris-setosa',value=1)).reshape((len(data['Class']),1))
y2=np.array(y2.replace(to_replace='Iris-versicolor',value=1)).reshape((len(data['Class']),1))
y3=np.array(y3.replace(to_replace='Iris-virginica',value=1)).reshape((len(data['Class']),1))
x=np.concatenate((np.ones((len(data['Class']),1)),np.array(inputdata)),axis=1)

# Initialize the parameters 
theta1=np.array([0.5,0.5,0.5,0.5,0.5]).reshape((5,1))
theta2=np.array([0.5,0.5,0.5,0.5,0.5]).reshape((5,1))
theta3=np.array([0.5,0.5,0.5,0.5,0.5]).reshape((5,1))

# Optimize all three parameter matrix for all three cases
theta1=gradient_descent(x,y1,theta1,alpha)
theta2=gradient_descent(x,y2,theta2,alpha)
theta3=gradient_descent(x,y3,theta3,alpha)

# Plot the data and correlation matrix to check for dependencies
data.plot()
print (data.corr())

# Test the model on a test set
test = np.array([1,6.2,3.4,5.4,2.3]).reshape((1,5))
predict(test,theta1,theta2,theta3)
