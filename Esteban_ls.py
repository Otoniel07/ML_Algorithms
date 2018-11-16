import numpy as np
from matplotlib import style
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


data = np.loadtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = data.shape
data = np.c_[np.ones(n),data]

# training sample and label 
sample_train = sample[0:30,:]
label_train = label[0:30]
# testing sample and label 
sample_test = sample[30:-1,:]
label_test = label[30:-1]

# Getting Transpose Matrices
sample_train_trans = np.transpose(sample_train)
sample_train_trans
label_train_trans = np.transpose(label_train)

# Collecting the AA Community and Non-AA Community
aa_sample_train = np.where(sample_train[:,3] >= .5)[0]
non_aa_sample_train = np.where(sample_train[:,3] < .5)[0]
aa_sample_test = np.where(sample_test[:,3] >= .5)[0]
non_aa_sample_test = np.where(sample_test[:,3] < .5)[0]

# Calculating Beta
betaXTX = np.matmul(sample_train_trans,sample_train)
betaXTXI = np.linalg.inv(betaXTX)
betaXTXIXT = np.matmul(betaXTXI,sample_train_trans)
beta = np.matmul(betaXTXIXT,label_train)

# Calculating Estimated Test Label and Total Population MSE
label_pred = np.matmul(sample_test,beta)
mse = mean_squared_error(label_test,label_pred)
aa_mse = np.sum((label_test[aa_sample_test] - label_pred[aa_sample_test])**2)/len(aa_sample_test) 
non_aa_mse = np.sum((label_test[non_aa_sample_test] - label_pred[non_aa_sample_test])**2)/len(non_aa_sample_test)

print("Total MSE: ", mse)
print("AA MSE: ", aa_mse)
print("Non-AA MSE: ", non_aa_mse)
