import numpy as np
import array
import math
from scipy.stats import multivariate_normal

data = np.genfromtxt('CrimeCommunityBinary.csv', delimiter=',')
#data = np.delete(data, 0, 1)
[n,p] = data.shape

test_size = 500

for j in [30, 100, 500, 1000]:
    
    train_size = j
    
    train_sample = data[0:train_size,0:-1]
    train_label = data[0:train_size,-1].reshape(train_size,1)
    test_sample = data[-500:,0:-1]
    test_label = data[-500:,-1].reshape(500,1)
    
    ones_sum = np.sum(train_label)
    zeros_sum = train_label.shape[0] - np.sum(train_label)
    ones_sum_test = np.sum(test_label)
    zeros_sum_test = test_label.shape[0] - np.sum(test_label)
    
    ones_sample = train_sample[np.where(train_label==1)[0]]
    zeros_sample = train_sample[np.where(train_label==0)[0]]
    
    pyc1 = ones_sum/train_size
    pyc0 = zeros_sum/train_size
    
    x0_sum = np.sum(zeros_sample, axis=0)
    x1_sum = np.sum(ones_sample, axis=0)
    
    mu0 = (1/zeros_sum)*x0_sum
    mu0 = np.expand_dims(mu0, axis=0)
    mu0 = np.transpose(mu0)
    mu1 = (1/ones_sum)*x1_sum
    mu1 = np.expand_dims(mu1, axis=0)
    mu1 = np.transpose(mu1)
    
    x1 = np.transpose(ones_sample[0, :])
    x1 = np.expand_dims(x1, axis=1)
    x1 = x1 - mu1
    x1T = np.transpose(x1)
    x1_sum_matrix = np.matmul(x1, x1T)
    for i in range(1, ones_sample.shape[0]):
        x1 = np.transpose(ones_sample[i, :])
        x1 = np.expand_dims(x1, axis=1)
        x1 = x1 - mu1
        x1T = np.transpose(x1)
        x1_sum_matrix = x1_sum_matrix + np.matmul(x1, x1T)
    
    x0 = np.transpose(zeros_sample[0, :])
    x0 = np.expand_dims(x0, axis=1)
    x0 = x0 - mu0
    x0T = np.transpose(x0)
    x0_sum_matrix = np.matmul(x0, x0T)
    for i in range(1, zeros_sample.shape[0]):
        x0 = np.transpose(zeros_sample[i, :])
        x0 = np.expand_dims(x0, axis=1)
        x0 = x0 - mu0
        x0T = np.transpose(x0)
        x0_sum_matrix = x0_sum_matrix + np.matmul(x0, x0T)
    
    
    cov_QDA1 = 1/(ones_sum - 1)*x1_sum_matrix
    cov_QDA0 = 1/(zeros_sum - 1)*x0_sum_matrix
    cov_LDA = (2/(train_size - 2))*(x1_sum_matrix + x0_sum_matrix)
    
    for i in [0, .5, 1]:
        coef = i
        new_cov1 = coef*cov_QDA1 + (1 - coef)*cov_LDA
        new_cov0 = coef*cov_QDA0 + (1 - coef)*cov_LDA
        
        # I couldn't make the Normal Distribution Equation work so I used the built in normal distribution function
        
        #l = 1e-2
        #XT_NCI_X_1 = np.matmul(np.matmul(test_sample, np.linalg.inv(new_cov1 + (1e-5)*np.identity(100))), np.transpose(test_sample))
        #slogdet = np.linalg.slogdet(new_cov1)
        #new_cov_det = np.exp(slogdet[1])
        #pp = (1/((math.sqrt(2*l)**2)*.0001))*np.exp((-1/2)*np.diag(XT_NCI_X_1))
        #
        #XT_NCI_X_0 = np.matmul(np.matmul(test_sample, np.linalg.inv(new_cov1 + (1e-5)*np.identity(100))), np.transpose(test_sample))
        #slogdet = np.linalg.slogdet(new_cov0)
        #new_cov_det = np.exp(slogdet[1])
        #pp = (1/((math.sqrt(2*l)**2)*np.linalg.det(new_cov0)))*np.exp((-1/2)*np.diag(XT_NCI_X_0))
        
        
        mu0 = (1/zeros_sum)*x0_sum
        mu0 = np.transpose(mu0)
        mu1 = (1/ones_sum)*x1_sum
        mu1 = np.transpose(mu1)
        
        normal0 = multivariate_normal.pdf(test_sample, mean=mu0, cov=new_cov0, allow_singular=True) 
        normal1 = multivariate_normal.pdf(test_sample, mean=mu1, cov=new_cov1, allow_singular=True)
        
        probability0 = normal0 * pyc0
        probability1 = normal1 * pyc1
        
        label_pred = np.zeros((len(test_label),1)) # initialize all decisions to class 0
        index_pred_c1 = np.where(probability1 > probability0) # update those predicted to class 1
        label_pred[index_pred_c1] = 1
        error = (np.sum(label_pred!=test_label) / len(test_label))*100 # evaluate classification error (i.e. count number of mis-classified instances)
        
        print("\n Error = %.2f" % (error))
        

