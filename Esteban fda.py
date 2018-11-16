import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('YaleFace.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]

mu = np.mean(sample,axis=0)
n = sample.shape[0]
l = 2

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

mu_c1 = np.mean(sample[index_c1,:],axis=0)
mu_c2 = np.mean(sample[index_c2,:],axis=0)
mu_c3 = np.mean(sample[index_c3,:],axis=0)
mu_c4 = np.mean(sample[index_c4,:],axis=0)
mu_c5 = np.mean(sample[index_c5,:],axis=0)
mu_c6 = np.mean(sample[index_c6,:],axis=0)
mu_c7 = np.mean(sample[index_c7,:],axis=0)
mu_c8 = np.mean(sample[index_c8,:],axis=0)
mu_c9 = np.mean(sample[index_c9,:],axis=0)
mu_c10 = np.mean(sample[index_c10,:],axis=0)
mu_c11 = np.mean(sample[index_c11,:],axis=0)
mu_c12 = np.mean(sample[index_c12,:],axis=0)
mu_c13 = np.mean(sample[index_c13,:],axis=0)
mu_c14 = np.mean(sample[index_c14,:],axis=0)
mu_c15 = np.mean(sample[index_c15,:],axis=0)
        
sigma_w_c1 = np.dot( (sample[index_c1,:] - mu_c1).transpose(), (sample[index_c1,:] - mu_c1) )
sigma_w_c2 = np.dot( (sample[index_c2,:] - mu_c2).transpose(), (sample[index_c2,:] - mu_c2) )
sigma_w_c3 = np.dot( (sample[index_c3,:] - mu_c3).transpose(), (sample[index_c3,:] - mu_c3) )
sigma_w_c4 = np.dot( (sample[index_c4,:] - mu_c4).transpose(), (sample[index_c4,:] - mu_c4) )
sigma_w_c5 = np.dot( (sample[index_c5,:] - mu_c5).transpose(), (sample[index_c5,:] - mu_c5) )
sigma_w_c6 = np.dot( (sample[index_c6,:] - mu_c6).transpose(), (sample[index_c6,:] - mu_c6) )
sigma_w_c7 = np.dot( (sample[index_c7,:] - mu_c7).transpose(), (sample[index_c7,:] - mu_c7) )
sigma_w_c8 = np.dot( (sample[index_c8,:] - mu_c8).transpose(), (sample[index_c8,:] - mu_c8) )
sigma_w_c9 = np.dot( (sample[index_c9,:] - mu_c9).transpose(), (sample[index_c9,:] - mu_c9) )
sigma_w_c10 = np.dot( (sample[index_c10,:] - mu_c10).transpose(), (sample[index_c10,:] - mu_c10) )
sigma_w_c11 = np.dot( (sample[index_c11,:] - mu_c11).transpose(), (sample[index_c11,:] - mu_c11) )
sigma_w_c12 = np.dot( (sample[index_c12,:] - mu_c12).transpose(), (sample[index_c12,:] - mu_c12) )
sigma_w_c13 = np.dot( (sample[index_c13,:] - mu_c13).transpose(), (sample[index_c13,:] - mu_c13) )
sigma_w_c14 = np.dot( (sample[index_c14,:] - mu_c14).transpose(), (sample[index_c14,:] - mu_c14) )
sigma_w_c15 = np.dot( (sample[index_c15,:] - mu_c15).transpose(), (sample[index_c15,:] - mu_c15) )


sigma_b_c1 = np.dot(np.expand_dims((mu_c1 - mu),0).transpose(), np.expand_dims((mu_c1 - mu),0))
sigma_b_c2 = np.dot(np.expand_dims((mu_c2 - mu),0).transpose(), np.expand_dims((mu_c2 - mu),0))
sigma_b_c3 = np.dot(np.expand_dims((mu_c3 - mu),0).transpose(), np.expand_dims((mu_c3 - mu),0))
sigma_b_c4 = np.dot(np.expand_dims((mu_c4 - mu),0).transpose(), np.expand_dims((mu_c4 - mu),0))
sigma_b_c5 = np.dot(np.expand_dims((mu_c5 - mu),0).transpose(), np.expand_dims((mu_c5 - mu),0))
sigma_b_c6 = np.dot(np.expand_dims((mu_c6 - mu),0).transpose(), np.expand_dims((mu_c6 - mu),0))
sigma_b_c7 = np.dot(np.expand_dims((mu_c7 - mu),0).transpose(), np.expand_dims((mu_c7 - mu),0))
sigma_b_c8 = np.dot(np.expand_dims((mu_c8 - mu),0).transpose(), np.expand_dims((mu_c8 - mu),0))
sigma_b_c9 = np.dot(np.expand_dims((mu_c9 - mu),0).transpose(), np.expand_dims((mu_c9 - mu),0))
sigma_b_c10 = np.dot(np.expand_dims((mu_c10 - mu),0).transpose(), np.expand_dims((mu_c10 - mu),0))
sigma_b_c11 = np.dot(np.expand_dims((mu_c11 - mu),0).transpose(), np.expand_dims((mu_c11 - mu),0))
sigma_b_c12 = np.dot(np.expand_dims((mu_c12 - mu),0).transpose(), np.expand_dims((mu_c12 - mu),0))
sigma_b_c13 = np.dot(np.expand_dims((mu_c13 - mu),0).transpose(), np.expand_dims((mu_c13 - mu),0))
sigma_b_c14 = np.dot(np.expand_dims((mu_c14 - mu),0).transpose(), np.expand_dims((mu_c14 - mu),0))
sigma_b_c15 = np.dot(np.expand_dims((mu_c15 - mu),0).transpose(), np.expand_dims((mu_c15 - mu),0))

sigma_w = (sigma_w_c1 + sigma_w_c2 + sigma_w_c3 + sigma_w_c4 + sigma_w_c5 + 
           sigma_w_c6 + sigma_w_c7 + sigma_w_c8 + sigma_w_c9 + sigma_w_c10 + 
           sigma_w_c11 + sigma_w_c12 + sigma_w_c13 + sigma_w_c14 + sigma_w_c15)
    
sigma_b = (sigma_b_c1 + sigma_b_c2 + sigma_b_c3 + sigma_b_c4 + sigma_b_c5 + 
           sigma_b_c6 + sigma_b_c7 + sigma_b_c8 + sigma_b_c9 + sigma_b_c10 + 
           sigma_b_c11 + sigma_b_c12 + sigma_b_c13 + sigma_b_c14 + sigma_b_c15)

sigma = sigma_w - (l*sigma_b)

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