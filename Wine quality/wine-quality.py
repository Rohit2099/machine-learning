# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:30:19 2018

@author: Rohit
"""

import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix

sns.set()

#Load the data and rename the columns
redw = pd.read_csv("winequality-red.csv",sep=';', header=0,dtype='float')
redw['wine']="redwine"

whitew = pd.read_csv("winequality-white.csv", sep=';', header=0,dtype='float')
whitew['wine']="whitewine"


data=pd.concat([redw,whitew],axis=0)
x=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality","wine"]
data.columns=columns

#Get the heat map matrix to check the correlation between different attributes
sns.heatmap(data.corr())

#Since we see there are a few dark spots on the heat map, we use feature selection to
#optimize the data


# Performing feature selection using chi square statistic
fclf=SelectKBest(score_func=chi2,k=7)
feature=fclf.fit(data[x],data['wine'])
print (feature.scores_)
newdata=feature.transform(data[x])

#Train the new data using the SVM model 
xtrain,xtest,ytrain,ytest=train_test_split(newdata,data['wine'],random_state=5)
clf=svm.SVC(kernel='linear',C=10,random_state=5)
clf.fit(xtrain,ytrain)
result=clf.predict(xtest)

# Check the accuracy using the confusion matrix and accuracy score
print ('Accuracy of the Model : ',accuracy_score(ytest,result))
print ('Confusion Matrix : ')
print (confusion_matrix(result,ytest))
