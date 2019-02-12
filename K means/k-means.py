# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:58:30 2018

@author: Rohit

"""

# Image compression using K means

import numpy as np
from PIL import Image
from skimage import io
import scipy.misc

# Computes the location of centroid of all clusters
def computeCentroids(x,k,pos):
    centroids=np.zeros((k,np.size(x,1)))
    for i in range(k):         
        ci = pos==i
        ci = ci.astype(int)
        total_number = np.sum(ci);
        ci=ci.reshape((np.size(x,0),1))
        total_matrix = np.matlib.repmat(ci,1,np.size(x,1))
        ci = np.transpose(ci)
        total = np.multiply(x,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

# Assigns the cluster number for all data points
def assignCluster(x,k,pos):
    idx = np.zeros((np.size(x,0),1))
    arr = np.empty((np.size(x,0),1))
    for i in range(k):
        y = pos[i]
        temp = np.ones((np.size(x,0),1))*y
        b = np.power((x-temp),2)
        a = np.sum(b,axis = 1)
        a=a.reshape((np.size(x,0),1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

# Performs feature scaling on the R,G,B pixel values(features)        
def featureScaling(x):
    mean_matrix = np.mean(x, axis = 0)
    std_matrix = np.std(x, axis = 0)
    res_matrix = (x-mean_matrix)/std_matrix
    return res_matrix

# Randomly initialize the centroids of k clusters to k differnt data points
def initializeK(x,k):
    k_idx = np.random.randint(0,len(x)-1,size=k)
    k_centroids=np.array(x[k_idx])
    return k_centroids

# Computes the eucledian dstance between two data points
def distance(x1,x2):
    diff=x1-x2
    diffsq=diff**2
    sumrows=np.sum(diffsq)
    dist=np.sqrt(sumrows)
    return dist

# Runs the k-means algorithm for a certain number of iteration count
def run_kmeans(iter_cnt,k,x,k_clusters):
    for i in range(iter_cnt):    
        x_clusters=assignCluster(x,k_clusters,k)
        k=computeCentroids(x,k_clusters,x_clusters)
    return x_clusters,k

# Load and display the image 
img = Image.open('64969496_p0_master1200.jpg')
color_matrix = np.array(img)
img.show()

# Define different constants and parameters
color_dim = color_matrix.shape
rows = color_dim[0]
cols = color_dim[1]
x = color_matrix.reshape(rows*cols,3)
m = x.shape
k_clusters=16
iter_cnt=2

# x_clusters contains the information about the cluster number of each data point
x_clusters=np.ones((np.size(x,0),1))

# Initialize the k centroids
k=initializeK(x,k_clusters)
# Train the model
x_clusters,k=run_kmeans(iter_cnt,k,x,k_clusters)

# Reassign the values of input image RGB matrix such that
# all the pixel values are replaced by its closest centroid
X_recovered = k[x_clusters]
# Reshape the matrix to a RGB matrix
X_recovered = np.reshape(X_recovered, (rows, cols, 3))

# Save the compressed image and display it
scipy.misc.imsave('tiger_compressed.jpg', X_recovered)
image_compressed = io.imread('tiger_small.jpg')
io.imshow(image_compressed)
io.show()
