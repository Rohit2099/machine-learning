# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 23:43:21 2018

@author: Rohit
"""
# Simple linear Regression using the iterative approach
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

sb.set()

# Calculates the loss due to the current parameters
def getErr(x,y,th):
    thTx=x.dot(th)
    err=0
    for i in range(len(x)):
        err+=(thTx[i][0]-y[i][0])**2
    err*=1/(2*len(x))
    return err

# Outputs the gradient parameters
def getderive(x,y,th,idx):
    thtx=x.dot(th)
    sum=0
    for i in range(len(x)):
        sum+=(thtx[i][0]-y[i][0])*x[i][idx]
       # print(str(i)+"-"+str(sum)+"-"+str(x[i][idx]))
    return sum    

# Performs the gradient descent algorithm
def gradient_descent(Newx,y,theta,m,epsilon):
    count=0
    i=0
    while i<500:
        error=getErr(Newx, y, theta)
        th0 = theta[0][0]-(alpha*getderive(Newx,y,theta,0))/m
        th1 = theta[1][0]-(alpha*getderive(Newx,y,theta,1))/m
        theta[0][0],theta[1][0] = th0,th1
        count=count+1
        error1=getErr(Newx, y, theta)  
        errorGraph.append(error1)
        i=i+1
        print(str(count) + "st iteration error -- " + str(error1))    
        if error<error1:
            print("Too large alpha")
            break
        if error-error1<epsilon:
            break
    return

# Predict the output with given theta
# Make adjustments on X to fit the model
def predict(inputX,outputY,theta):
    inputZ = np.ones((len(x),1))
    newInputX=np.concatenate((inputZ,inputX),axis=1)
    predictedY=newInputX.dot(theta)
    return predictedY

# Load the data
data = pd.read_csv("train.csv")

# Clean the data and resize it. Add a column of 1s for easier calculation
x = np.array(data['x']).reshape((len(data['x']),1))
Newx = np.concatenate((np.ones((len(x),1)),x),axis=1)
y = np.array(data['y']).reshape((len(data['y']),1))

# Initialize the parameters
theta = np.ones((2,1))         
thetaTrans = theta.T

error = getErr(Newx,y,theta)
# Define all constants and hyper parameters
epsilon =0.000001
alpha=0.0005
m = len(x)
# Initialize a list to plot the error vs. iteration graph
errorGraph=[]

# Train the model
gradient_descent(Newx,y,theta,m,epsilon)

# Plot the error graph
f=plt.figure(1)
plt.plot(errorGraph,c='orange',linestyle='--')
plt.xlabel("No. of iterations ")
plt.ylabel("error")
plt.title("error check")

# Load the test set 
train=pd.read_csv("test.csv")
inputX=np.array(data['x']).reshape((len(data['x']),1))
outputY=np.array(data['y']).reshape((len(data['y']),1))

# Get the predicted output
predictedY=predict(inputX,outputY,theta)

# Plot the predicted outpput and the real output 
# to check the corectness
h=plt.figure(2)
plt.plot(inputX,predictedY,c='r',linewidth=5)
plt.scatter(inputX,outputY,c='black')
plt.xlabel("inputX")
plt.ylabel("PredictedY and givenY")
plt.title("Predicted vs Given")
