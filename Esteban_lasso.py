import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('crimerate.csv', delimiter=',')

# training sample and label 
sample_train = data[0:30,0:-1]
label_train = data[0:30,-1]
# testing sample and label 
sample_test = data[30:-1,0:-1]
label_test = data[30:-1,-1]
 
Beta = np.random.rand(100, 1)
MSE = 1000
lamda = 1e-4

MSE_list = []
iterations = []
non_zero_coef = []

i = 0
while(i <= 1000 and MSE > 1):
    j = np.random.randint(0, 100)

    sample_j = sample_train[:,j]
    sample_no_j = np.delete(sample_train, j, 1)
    Beta_no_j = np.delete(Beta, j, 0)
    A = np.matmul(sample_no_j, Beta_no_j)
    
    Sum_2XA = 2*np.dot(sample_j.transpose(),A)
    Sum_2X = 2*np.dot(sample_j.transpose(),sample_j)
    
    if(Sum_2XA < -lamda):        
        Beta[j] = (-lamda - Sum_2XA)/Sum_2X    
    elif(Sum_2XA > lamda):
        Beta[j] = (lamda - Sum_2XA)/Sum_2X
    else:
        Beta[j] = 0

    label_pred = np.squeeze(np.dot(sample_train, Beta))
    MSE = np.sum((label_train-label_pred)**2)/len(label_train)
    
    MSE_list = MSE_list + [MSE]
    iterations = iterations + [i]
    non_zero_coef = non_zero_coef + [sum(Beta != 0)]
    
    i = i + 1


# Collecting the AA Community and Non-AA Community
index_AA = np.where(sample_test[:,2] >= .5)[0]
index_nAA = np.where(sample_test[:,2] < .5)[0]        

# Calculating Estimated Test Label and Population MSE

label_pred = np.squeeze(np.dot(sample_test, Beta))
MSE_tot =  np.sum((label_test-label_pred)**2)/len(label_test)
print("Total MSE: ", MSE_tot)

MSE_AA = np.sum((label_test[index_AA]-label_pred[index_AA])**2)/len(label_test[index_AA])
print("AA MSE: ", MSE_AA)

MSE_nAA = np.sum((label_test[index_nAA]-label_pred[index_nAA])**2)/len(label_test[index_nAA])
print("Non-AA MSE: ", MSE_nAA)

plt.plot(iterations, MSE_list)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()   
 
plt.plot(iterations, non_zero_coef)
plt.xlabel('Iterations')
plt.ylabel('non-zero regression coefficients')
plt.show()