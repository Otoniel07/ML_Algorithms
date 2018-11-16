import numpy as np
from sklearn import tree


data = np.loadtxt('CrimeCommunityBinary_corrected.csv', delimiter=',')
[n,p] = data.shape
data = np.c_[np.ones(n),data]

# training sample and label 
sample = data[0:500]
sample_train = data[0:500,0:-1]
label_train = data[0:500,-1]
# testing sample and label 
sample_test = data[500:-1,0:-1]
label_test = data[500:-1,-1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(sample_train, label_train)
decision_tree_pred = clf.predict(sample_test)

index_correct = np.where(decision_tree_pred==label_test)[0]
err1 = ((1492-index_correct.shape[0])/1492)*100
print('\nDecision Tree err = %f' % err1)


bootstrap_sum = np.zeros(1492)
m = 5
for i in range(m):
    bootstrap_sample = sample[np.random.randint(sample.shape[0], size=100), :]
    clf = clf.fit(bootstrap_sample[0:100,0:-1], bootstrap_sample[0:100,-1])
    bootstrap_pred = clf.predict(sample_test)
    bootstrap_sum = np.add(bootstrap_sum, bootstrap_pred)

bag_ave = np.zeros(1492)
for j in range(1492):
    if bootstrap_sum[j] < 3:
        bag_ave[j] = 0
    else:
        bag_ave[j] = 1
        
index_correct2 = np.where(bag_ave==label_test)[0]
err2 = ((1492-index_correct2.shape[0])/1492)*100
print('\nBagged Tree err = %f' % err2)