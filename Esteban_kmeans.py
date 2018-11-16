import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt('YaleFace.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]

cycles = 30
    
#for K in ([3,5,7,9]):
K = 3
mu_k = sample[np.random.choice(sample.shape[0], K, replace=True), :]
sum_k = np.zeros((sample.shape[0], K))
distortion = 0
distortion_array = np.zeros(cycles)

for c in range(cycles):
    for i in range(K):
        sum_k[:,i] = np.sum(np.power(sample - mu_k[i], 2), axis=1)
        
    sum_k_sorted = np.argsort(sum_k)
    sample_clustered = sum_k_sorted[:,0]
    
    for i in range(K):
        index_c = np.where(sample_clustered==i)[0]
        distortion = np.sum(np.power(sample - mu_k[i], 2), axis=1).sum()
        distortion_array[c] = distortion_array[c] + distortion
        mu_k[i] = np.mean(sample[index_c,:], axis=0)
        
plt.plot(distortion_array[0:cycles], color='b')
print(distortion_array[29])

K = 5
mu_k = sample[np.random.choice(sample.shape[0], K, replace=True), :]
sum_k = np.zeros((sample.shape[0], K))
distortion = 0
distortion_array = np.zeros(cycles)
for c in range(cycles):
    for i in range(K):
        sum_k[:,i] = np.sum(np.power(sample - mu_k[i], 2), axis=1)
        
    sum_k_sorted = np.argsort(sum_k)
    sample_clustered = sum_k_sorted[:,0]
    
    for i in range(K):
        index_c = np.where(sample_clustered==i)[0]
        distortion = np.sum(np.power(sample - mu_k[i], 2), axis=1).sum()
        distortion_array[c] = distortion_array[c] + distortion
        mu_k[i] = np.mean(sample[index_c,:], axis=0)
        
plt.plot(distortion_array[0:cycles], color='r')
print(distortion_array[29])

K = 7
mu_k = sample[np.random.choice(sample.shape[0], K, replace=True), :]
sum_k = np.zeros((sample.shape[0], K))
distortion = 0
distortion_array = np.zeros(cycles)
for c in range(cycles):
    for i in range(K):
        sum_k[:,i] = np.sum(np.power(sample - mu_k[i], 2), axis=1)
        
    sum_k_sorted = np.argsort(sum_k)
    sample_clustered = sum_k_sorted[:,0]
    
    for i in range(K):
        index_c = np.where(sample_clustered==i)[0]
        distortion = np.sum(np.power(sample - mu_k[i], 2), axis=1).sum()
        distortion_array[c] = distortion_array[c] + distortion
        mu_k[i] = np.mean(sample[index_c,:], axis=0)
        
plt.plot(distortion_array[0:cycles], color='g')
print(distortion_array[29])

K = 9   
mu_k = sample[np.random.choice(sample.shape[0], K, replace=True), :]
sum_k = np.zeros((sample.shape[0], K))
distortion = 0
distortion_array = np.zeros(cycles) 
for c in range(cycles):
    for i in range(K):
        sum_k[:,i] = np.sum(np.power(sample - mu_k[i], 2), axis=1)
        
    sum_k_sorted = np.argsort(sum_k)
    sample_clustered = sum_k_sorted[:,0]
    
    for i in range(K):
        index_c = np.where(sample_clustered==i)[0]
        distortion = np.sum(np.power(sample - mu_k[i], 2), axis=1).sum()
        distortion_array[c] = distortion_array[c] + distortion
        mu_k[i] = np.mean(sample[index_c,:], axis=0)
        
plt.plot(distortion_array[0:cycles], color='y')    
print(distortion_array[29])