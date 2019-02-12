# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:40:40 2018

@author: Rohit
"""
# Import all the necessary packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer
import re
import nltk  
nltk.download('wordnet')  
nltk.download('snowball')  
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

def stem(data,idx):
    stemmer = SnowballStemmer('english')
    for sen in range(8):
        #print(data[idx][sen])
        document = data[idx][sen].split()
        document = [stemmer.stem(word) for word in document]
        document = ' '.join(document)
        #print (document)
        data[idx][sen]=document
    return data


def lemmatize(data,idx):
    lmtizr = WordNetLemmatizer()
    for sen in range(8):
        #print(data[idx][sen])
        document = data[idx][sen].split()
        document = [lmtizr.lemmatize(word) for word in document]
        document = ' '.join(document)
        #print (document)
        data[idx][sen]=document
    return data



def clean_data(data,idx):
    print(len(data[idx][0]))
    for sen in range(len(data)):
        data[idx][sen].lower()
        
        data[idx][sen]=re.sub('^(\s|^\w)+','',data[idx][sen])
    return data

# Load the data and sample it 
data=pd.read_csv('train.csv',header=0,encoding="ISO-8859-1")
test=pd.read_csv('test.csv',header=0,encoding="ISO-8859-1")


# Rename the columns and clean the data
columns=['Item ID','Sentiment','Text']
testcolumns=['Item ID','Text']
data=data.dropna(inplace=False)
data.columns=columns
test.columns=testcolumns

# Clean the data
# Takes toooo long to clean the data
#data=clean_data(data,'Text')

# Lemmatize the documents
#data=lemmatize(data,'Text')
#lmtizr=WordNetLemmatizer()

# Stemming the documents
data=stem(data,'Text')

xtrain,xtest,ytrain,ytest=train_test_split(data['Text'],data['Sentiment'],random_state=5)
my_stopword_list = ['and','to','the','of','I','i',"I'm","i'm",'is']

# We'll use the bag of words structure to count the frequency
# min_df mentions the min number of occurence for it to be considered a feature
countVec=CountVectorizer(min_df=5,stop_words=my_stopword_list)
countVec.fit(xtrain)

# Get the feature/vocabulary by the following method
# Prints every 500th feature
feature_names=countVec.get_feature_names()
#print (countVec.get_feature_names()[::500])

# Now we calculate the frequency table
x_vec=countVec.transform(xtrain)

# Numerical Array form of the input text
x_trans=x_vec.toarray()


# Lets apply Logistic Regression to classify the positive feedback and negative ones
model=LogisticRegression()
model.fit(x_vec,ytrain)
predictions=model.predict(countVec.transform(xtest))


"""
# SVM
model_sv=SVC()
model_sv.fit(x_vec,ytrain)
predictions_sv=model_sv.predict(countVec.transform(xtest))
"""

# Check the accuracy using the area under the curve
print ('Count Vectorizer Logistic regression Accuracy:' + str(roc_auc_score(predictions,ytest)))
#print ('Count Vectorizer SVM Accuracy:' + str(roc_auc_score(predictions_sv,ytest)))

# Lets try another transform function
tf_x=TfidfVectorizer(min_df=5,ngram_range=(1,2),stop_words=my_stopword_list)
tf_x.fit(xtrain)

# Logistic Regression
model2=LogisticRegression()
model2.fit(tf_x.transform(xtrain),ytrain)
predictions_tf_lg=model2.predict(tf_x.transform(xtest))

# SVM
"""
model3=SVC(kernel='rbf')
model3.fit(tf_x.transform(xtrain),ytrain)
predictions_tf_svm=model2.predict(tf_x.transform(xtest))
"""
print('Tfidf Vectorizer Logistic Regression Accuracy:' + str(roc_auc_score(predictions_tf_lg,ytest)))
#print('Tfidf Vectorizer SVM Accuracy:' + str(roc_auc_score(predictions_tf_svm,ytest)))

hashVec=HashingVectorizer(n_features=20,stop_words=my_stopword_list)
hashVec.fit(xtrain)
hash_x=hashVec.transform(xtrain)

# Logistic Regression
model3=LogisticRegression()
model3.fit(hash_x,ytrain)
predictions_hash_lg=model3.predict(hashVec.transform(xtest))

print('Hashing Vectorizer Logistic Regression Accuracy:' + str(roc_auc_score(predictions_hash_lg,ytest)))

