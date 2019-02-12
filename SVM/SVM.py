# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 02:21:25 2018

@author: Rohit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sb

sb.set()

def svm_model(xtrain,ytrain):  
    postiveX=[]
    negativeX=[]
    xtr=np.array(xtrain).reshape((len(xtrain),2))
    ytr=np.array(ytrain).reshape((len(ytrain),1))
    for i,v in enumerate(ytr):
        if v==0:
            negativeX.append(xtr[i])
        else:
            postiveX.append(xtr[i])
    
    #our data dictionary
    data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)}
    max_feature=float('-inf')
    min_feature=float('+inf')
            
    for yi in data_dict:
        if np.amax(data_dict[yi])>max_feature:
            max_feature=np.amax(data_dict[yi])
                    
        if np.amin(data_dict[yi])<min_feature:
                min_feature=np.amin(data_dict[yi])
    
    learning_rate = [max_feature * 0.1, max_feature * 0.01, max_feature * 0.001]
    
    
    length_Wvector = {}
    transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
    
    b_step_size = 2
    b_multiple = 5
    w_optimum = max_feature*10

    for lrate in learning_rate:
        
        w = np.array([w_optimum,w_optimum])     
        optimized = False
        while not optimized:
            for b in np.arange(-1*(max_feature * b_step_size), max_feature * b_step_size, lrate*b_multiple):
                for transformation in transforms:  
                    wtrans = w*transformation
                    
                    correctly_classified = True
                    
                    for yi in data_dict:
                        for xi in data_dict[yi]:
                            if yi*(np.dot(wtrans,xi)+b) < 1:  
                                correctly_classified = False
                                
                    if correctly_classified:
                        length_Wvector[np.linalg.norm(wtrans)] = [wtrans,b] 
            if w[0] < 0:
                optimized = True
            else:
                w = w - lrate

        norms = sorted([n for n in length_Wvector])
        minimum_wlength = length_Wvector[norms[0]]
        w = minimum_wlength[0]
        b = minimum_wlength[1]
        w_optimum = w[0]+lrate*2
    return (w,b)

# Predicts using the svm model
def predict(x,w,b):
    temp=np.dot(x,np.array(w).T)+b
    if temp<0:
        return 0
    else:
        return 1
    
    
# Generate the data and split into test and train
samplec=70
(x,y) =  make_blobs(n_samples=samplec,n_features=2,centers=2,cluster_std=1.05,random_state=40)



xtrain=x[0:45,:]
ytrain=y[0:45]

xtest=x[45:,:]
ytest=y[45:]


# Y=W.X+B 
w=[] 
b=[] 

w,b=svm_model(xtrain,ytrain)

for i in range(25):
    print ("Prediction "+str(i+1))   
    print (predict(xtest[i],w,b))
    print (ytest[i])

# Test the model

X=xtrain[:,0]
Y=(-w[0]*X-b)/w[1]

# Visualize the model
plt.scatter(xtrain[:,0],xtrain[:,1],color='orange')
plt.plot(X,Y,color='green')
plt.show()
