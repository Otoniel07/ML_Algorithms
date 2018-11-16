import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('YaleFace.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]

index_c1 = np.where(label==1)[0]
index_c2 = np.where(label==2)[0]
index_c3 = np.where(label==3)[0]
index_c4 = np.where(label==4)[0]
index_c5 = np.where(label==5)[0]
index_c6 = np.where(label==6)[0]
index_c7 = np.where(label==7)[0]
index_c8 = np.where(label==8)[0]
index_c9 = np.where(label==9)[0]
index_c10 = np.where(label==10)[0]
index_c11 = np.where(label==11)[0]
index_c12 = np.where(label==12)[0]
index_c13 = np.where(label==13)[0]
index_c14 = np.where(label==14)[0]
index_c15 = np.where(label==15)[0]

mu = np.mean(sample,axis=0)
n = sample.shape[0]

sigma = np.dot( (sample[:,:] - mu).transpose(), (sample[:,:] - mu) ) / n

w, v = np.linalg.eig(sigma)

w_desc = np.argsort(w)[::-1]

w1 = w[0].real
v1 = v[:,0].real
w2 = w[1].real
v2 = v[:,1].real

plt.figure()

plt.scatter(np.dot(sample[index_c1,:], v1),np.dot(sample[index_c1,:], v2))
plt.scatter(np.dot(sample[index_c2,:], v1),np.dot(sample[index_c2,:], v2))
plt.scatter(np.dot(sample[index_c3,:], v1),np.dot(sample[index_c3,:], v2))
plt.scatter(np.dot(sample[index_c4,:], v1),np.dot(sample[index_c4,:], v2))
plt.scatter(np.dot(sample[index_c5,:], v1),np.dot(sample[index_c5,:], v2))
plt.scatter(np.dot(sample[index_c6,:], v1),np.dot(sample[index_c6,:], v2))
plt.scatter(np.dot(sample[index_c7,:], v1),np.dot(sample[index_c7,:], v2))
plt.scatter(np.dot(sample[index_c8,:], v1),np.dot(sample[index_c8,:], v2))
plt.scatter(np.dot(sample[index_c9,:], v1),np.dot(sample[index_c9,:], v2))
plt.scatter(np.dot(sample[index_c10,:], v1),np.dot(sample[index_c10,:], v2))
plt.scatter(np.dot(sample[index_c11,:], v1),np.dot(sample[index_c11,:], v2))
plt.scatter(np.dot(sample[index_c12,:], v1),np.dot(sample[index_c12,:], v2))
plt.scatter(np.dot(sample[index_c13,:], v1),np.dot(sample[index_c13,:], v2))
plt.scatter(np.dot(sample[index_c14,:], v1),np.dot(sample[index_c14,:], v2))
plt.scatter(np.dot(sample[index_c15,:], v1),np.dot(sample[index_c15,:], v2))

plt.show()