import numpy as np
from matplotlib import style
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances


data = np.loadtxt('studentgrade.csv', delimiter=',')
[n,p] = data.shape
data = np.c_[np.ones(n),data]

# training sample and label 
sample_train = data[0:200,0:-1]
label_train = data[0:200,-1]
# testing sample and label 
sample_test = data[200:-1,0:-1]
label_test = data[200:-1,-1]

# Ridge Regression

lamda = 1e-1
Io = np.identity(p)
Io[0,0] = 0

# Getting Transpose Matrices
sample_train_trans = np.transpose(sample_train)
label_train_trans = np.transpose(label_train)

# Calculating Beta
betaXTX = np.dot(sample_train.transpose(), sample_train) + lamda*Io
betaXTXI = np.linalg.inv(betaXTX)
beta = np.dot(betaXTXI,sample_train.transpose()).dot(label_train)

# make prediction 
label_pred = np.dot(sample_test,beta) + lamda*np.transpose(beta).dot(Io).dot(beta)

# compute total MSE
mse = np.sum((label_test-label_pred)**2)/len(label_test)
print('\nMSE_RR = %f' % mse)


#Kernel Ridge Regression

gamma = 1e-9

#Training

K = np.exp(-gamma*euclidean_distances(sample_train, sample_train))
#alpha = (-1/lamda)*(np.dot(sample_test, beta) - label_test)
I = np.identity(sample_train.shape[0])
alpha = np.dot(np.linalg.inv(K + lamda*I), label_train)

label_pred_k = np.dot(K, alpha)

# compute total MSE
mse_k = np.sum((label_train-label_pred_k)**2)/len(label_train)
print('\nTraining_MSE_KRR = %f' % mse_k)

# Testing

K = np.exp(-gamma*euclidean_distances(sample_test, sample_test))
#alpha = (-1/lamda)*(np.dot(sample_test, beta) - label_test)
I = np.identity(sample_test.shape[0])
alpha = np.dot(np.linalg.inv(K + lamda*I), label_test)

label_pred_k = np.dot(K, alpha)

# compute total MSE
mse_k = np.sum((label_test-label_pred_k)**2)/len(label_test)
print('\nTesting_MSE_KRR = %f' % mse_k)


