import numpy as np

data = np.loadtxt('CrimeCommunityBinary.csv', delimiter=',')

[n,p] = data.shape

data = np.c_[np.ones(n),data]

train_size = 30

train_sample = data[0:train_size,0:-1]
train_label = data[0:train_size,-1]
test_sample = data[train_size:-1,0:-1]
test_label = data[train_size:-1,-1]

beta = np.ones(train_sample.shape[1])
beta = np.transpose(beta)
s = 0

for n in range(0, 20):
    p_vector = []
    for i in range(0, 30):
        a = np.array(train_sample).reshape((30, 101))
        x = a[i, :]
        xT = np.transpose(x)
        p_vector.append(1/(1 + np.exp(np.matmul(xT, beta))))
        
    p_array = np.transpose(np.asarray(p_vector))
    
    Y_p = np.subtract(train_label, p_array)
    #beta = [x * (1/s) for x in beta]
    L_prime = np.subtract(np.matmul(np.transpose(train_sample), Y_p), beta/s**2)
    
    identity_matrix = np.identity(30)
    identity_matrix2 = np.identity(101)
    identity_sigma = identity_matrix2/s**2
    pp = (1/(1 + np.exp(np.matmul(train_sample, beta))))*(1/(1 + np.exp(np.matmul(-train_sample, beta))))
    w_vector = identity_matrix*pp
    L_dprime = np.subtract(np.matmul(np.matmul(np.transpose(train_sample), w_vector), train_sample), identity_sigma)
    
    beta_subtract = np.matmul(np.linalg.inv(L_dprime), L_prime)
    beta = np.subtract(beta, beta_subtract)
    
    
label_pred = []
for y in range(0, test_label.shape[0]-1):
    if np.matmul(test_sample[y], beta) >= .5:
        label_pred.append(1)
    else:
        label_pred.append(0)

a = 0
for x in range(0, test_label.shape[0]-1):
    if label_pred[x] == test_label[x]:
        a = a + 1

accuracy = (1-(a/test_label.shape[0]))*100
print('Testing Error: {}%' .format(accuracy))