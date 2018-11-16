import numpy as np
import math


data1 = np.loadtxt('studentgrade1.csv', delimiter=',')
sample1 = data1[:,0:-1]
label1 = data1[:,-1]

data2 = np.loadtxt('studentgrade2.csv', delimiter=',')
sample2 = data2[:,0:-1]
label2 = data2[:,-1]

# Getting Transpose Matrices
sample1_trans = np.transpose(sample1)
label1_trans = np.transpose(label1)

sample2_trans = np.transpose(sample2)
label2_trans = np.transpose(label2)

shape1 = data1.shape
n1 = shape1[0]
data1_sum = np.sum(label1)


shape2 = data2.shape
n2 = shape2[0]
mu_sum1 = np.sum(sample1, axis=1)
mu_sum2 = np.sum(sample2, axis=1)
f = np.full((1,395), 45)
g = np.full((1,10), 45)
mu1 = mu_sum1 / g[None,:]
mu2 = mu_sum2 / f[None,:]

# Calculating Sigma1
sum1 = 0
for i in range(0,9):
    sum1 = sum1 + (label1[i] - (mu1[0,0,i]))**2
sigma1 = math.sqrt(sum1/n1)

# Calculating Sigma2
sum2 = 0
for j in range(0,394):
    sum2 = sum2 + (label2[j] - (mu2[0,0,j]))**2
sigma2 = math.sqrt(sum2/n2)

# Calculating MAP Estimators
map_sum1 = np.sum(label1) + 15
map_sum2 = np.sum(label2) + 15

map_estimator_1 = map_sum1/(n1 + (sigma1/2)**2)
map_estimator_2 = map_sum2/(n2 + (sigma2/2)**2)

map_estimator_3 = map_sum1/(n1 + (sigma1/(10**(-5))**2))
map_estimator_4 = map_sum2/(n2 + (sigma2/(10**(-5))**2))


print('\n MLE Estimator 1 = %f' % sigma2)
print('\n MAP Estimator 1 (σ2 = 2) = %f' % map_estimator_1)
print('\n MAP Estimator 2 (σ2 = 2) = %f' % map_estimator_2)
print('\n MAP Estimator 1 (σ2 = 10e-5) = %f' % map_estimator_1)
print('\n MAP Estimator 2 (σ2 = 10e-5) = %f' % map_estimator_2)


# Calculating Beta
#beta1XTX = np.matmul(sample1_trans,sample1)
#beta1XTXI = np.linalg.inv(beta1XTX)
#beta1XTXIXT = np.matmul(beta1XTXI,sample1_trans)
#beta1 = np.matmul(beta1XTXIXT,label1)
#
#beta2XTX = np.matmul(sample2_trans,sample2)
#beta2XTXI = np.linalg.inv(beta2XTX)
#beta2XTXIXT = np.matmul(beta2XTXI,sample2_trans)
#beta2 = np.matmul(beta2XTXIXT,label2)


#maps2 = np.divide(label2, n2 + (sigma1/2)**2)