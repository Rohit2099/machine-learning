# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:40:53 2018

@author: Rohit
"""

# A MLP with one hidden layer used to recognize hand-written digits
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Define the activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Define an alternative activation function
def tanh(z):
    temp1=np.exp(z)-np.exp(-z)
    temp2=np.exp(z)+np.exp(-z)
    return temp1/temp2

# Randomly initialize the parameters in the range -0.12 to +0.12.
# Keeps the algorithm effecient
def initialize_params():
    w1=np.random.randn(30,64)
    w2=np.random.randn(10,30)
    b1=np.random.randn(30,1)
    b2=np.random.randn(10,1)
    return w1,w2,b1,b2

# Predict the result which has the maximum probability after running run_MLP 
def predict(w1,b1,w2,b2,x):
    z1=np.dot(w1,x.T)+b1.T   
    a1=sigmoid(z1)
    z2=np.dot(w2,a1.T)+b2
    a2=sigmoid(z2)
    (m,i) = max((v,i) for i,v in enumerate(list(a2)))
    return i

# Runs the forward propogation by calculating the value of each neuron
def forward_propag(w1,w2,b1,b2,x,y):
    z1=np.dot(x,w1.T)+b1.T
    a1=sigmoid(z1)
    z2=np.dot(a1,w2.T)+b2.T
    a2=sigmoid(z2)
    cache={'z1':z1,'z2':z2,'a1':a1,'a2':a2}
    return cache

# Determines the error in the prediction with the current parameters
def cost_function(w1,w2,b1,b2,x,y):
    z1=np.dot(x,w1.T)+b1.T
    a1=sigmoid(z1)
    z2=np.dot(a1,w2.T)+b2.T
    a2=sigmoid(z2)
    temp=-(np.log10(a2)*y+(1-y)*np.log10((1-a2)))
    result=np.sum(temp)/sample_size
    return result

# Returns the gradient of sigmoid function
def sigmoid_gradient(z):
    temp=sigmoid(z)
    return temp*(1-temp)
    
# Computes the gradient of loss function with respect to 
# each parameter aiding the gradient descent
def back_propag(cache,x,y,w2):
    dz2=cache['a2']-y
    dw2=np.dot(dz2.T,cache['a1'])/sample_size
    db2=(np.sum(dz2,axis=0))/sample_size
    dz1=np.dot(dz2,w2)*sigmoid_gradient(cache['z1'])
    dw1=np.dot(dz1.T,x)/sample_size
    db1=(np.sum(dz1,axis=0))/sample_size
    db2=db2.reshape((10,1))
    db1=db1.reshape((30,1))
    cache2={'dw2':dw2,'dw1':dw1,'db2':db2,'db1':db1}
    return cache2

# This function runs the main MLP model to determine the correct 
# parameters to predict the output 
def run_MLP(w1,w2,b1,b2,x,y):   
    while True:
        cache=forward_propag(w1,w2,b1,b2,x,y)
        cache2=back_propag(cache,x,y,w2)
        w1=w1-alpha*cache2['dw1']
        w2=w2-alpha*cache2['dw2']
        b1=b1-alpha*cache2['db1']
        b2=b2-alpha*cache2['db2']
        error=cost_function(w1,w2,b1,b2,x,y)
        if error < epsilon:
            break
        print (error)
    return w1,w2,b1,b2

# Prints the accuracy of the model tested on a test set
def get_accuracy(xtest,w1,b1,w2,b2,ytest):
    accuracy=0
    for i in range(len(xtest)):
        xtest[i]=xtest[i].reshape((1,64))
        predicted=predict(w1,b1,w2,b2,xtest[i])
        print('Predicted value : ' + str(predicted))
        print('True value :     '+ str(ytest[i]))
        if predicted == ytest[i]:
            accuracy=accuracy+1
    acc_err=accuracy/len(xtest)
    print (acc_err*100)


# Initialize hyper parameters
alpha=0.3    
epsilon=0.06

# Load, clean and split the data
data=load_digits()
data.target=data.target.reshape(1797,1)
x,xtest,yold,ytest=train_test_split(data.data,data.target,test_size=0.2,random_state=42)
y=np.zeros((len(yold),10))

# Reshape the output column to aid the classification
for i in range(len(yold)):
    y[i][yold[i]]=1
    
sample_size=len(x)
    
#Initialize the parameters
w1,w2,b1,b2=initialize_params()

# Run the model
w1,w2,b1,b2=run_MLP(w1,w2,b1,b2,x,y)

# Get the accuracy
get_accuracy(xtest,w1,b1,w2,b2,ytest)


