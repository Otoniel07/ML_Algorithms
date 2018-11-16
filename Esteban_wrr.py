import numpy as np
from matplotlib import style
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = data.shape
data = np.c_[np.ones(n),data]

# training sample and label 
sample_train = data[0:30,0:-1]
label_train = data[0:30,-1]
# testing sample and label 
sample_test = data[30:-1,0:-1]
label_test = data[30:-1,-1]

# Getting Transpose Matrices
sample_train_trans = np.transpose(sample_train)
sample_train_trans
label_train_trans = np.transpose(label_train)


# Collecting the AA Community and Non-AA Community
aa_sample_train = np.where(sample_train[:,3] >= .5)[0]
non_aa_sample_train = np.where(sample_train[:,3] < .5)[0]
aa_sample_test = np.where(sample_test[:,3] >= .5)[0]
non_aa_sample_test = np.where(sample_test[:,3] < .5)[0]

# Constructing the weight matrix
weight_identities = np.ones(30)
weight_identities[aa_sample_train] = 1
weight_matrix = np.zeros((30,30))
np.fill_diagonal(weight_matrix, weight_identities)

lamda = 1e-2
Io = np.identity(p)
Io[0,0] = 0

# Calculating Beta
betaXTWX = np.matmul(np.matmul(sample_train_trans,weight_matrix),sample_train) + lamda*Io
betaXTWXI = np.linalg.inv(betaXTWX)
betaXTWXIXT = np.matmul(betaXTWXI,sample_train_trans)
betaXTWXIXTW = np.matmul(betaXTWXIXT,weight_matrix)
beta = np.matmul(betaXTWXIXTW,label_train)

# Predict Estimated Test Label and each MSE
label_pred = np.matmul(sample_test,beta) + lamda*np.transpose(beta).dot(Io).dot(beta)
mse = np.sum((label_test - label_pred)**2)/len(label_test)
aa_mse = np.sum((label_test[aa_sample_test] - label_pred[aa_sample_test])**2)/len(aa_sample_test) 
non_aa_mse = np.sum((label_test[non_aa_sample_test] - label_pred[non_aa_sample_test])**2)/len(non_aa_sample_test)

print("Total MSE: ", mse)
print("AA MSE: ", aa_mse)
print("Non-AA MSE: ", non_aa_mse)