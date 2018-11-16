import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('CrimeCommunityBinary_corrected.csv', delimiter=',')
[n,p] = data.shape
data = np.c_[np.ones(n),data]

# training sample and label 
sample_train = data[0:1500,0:-1]
label_train = data[0:1500,-1]
# testing sample and label 
sample_test = data[1500:-1,0:-1]
label_test = data[1500:-1,-1]


def perceptron_beta(beta, sample, label):
    n = 1
    
    for i, x in enumerate(sample):
        if (np.dot(sample[i], beta)*label[i]) <= 0:
            beta = beta + n*sample[i]*label[i]
    print(beta)            
    return beta




beta = np.zeros(len(sample_train[0]))
errors = []

for t in range(20):
    beta = perceptron_beta(beta, sample_train, label_train)
    
    test_pred = np.dot(sample_test, beta)
    label_pred = np.zeros(test_pred.shape[0])
    for i in range(test_pred.shape[0]):
        if test_pred[i] >= 0:
            label_pred[i] = 1
        else:
            label_pred[i] = 0
    
    index_err = np.where(label_pred!=label_test)[0]
    print(index_err.shape[0])
    errors.append(index_err.shape[0])
    
plt.plot(errors)
plt.xlabel('Trial')
plt.ylabel('Errors')