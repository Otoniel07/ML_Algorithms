import numpy as np
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier


data = np.loadtxt('CrimeCommunityBinary_corrected.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]

# training sample and label 
sample_train = sample[0:30,:]
label_train = label[0:30]
# testing sample and label 
sample_test = sample[30:-1,:]
label_test = label[30:-1]

# Linear SVM
clf = LinearSVC()
clf.fit(sample_train, label_train)

label_pred = clf.predict(sample_train)
j=0
for i in range(label_train.shape[0]):
    if label_train[i] == label_pred[i]:
        j = j+1

training_error = j/label_test.shape[0] * 100
print(training_error)


label_pred = clf.predict(sample_test)
j=0
for i in range(label_test.shape[0]):
    if label_test[i] == label_pred[i]:
        j = j+1

testing_error = j/label_test.shape[0] * 100
print(testing_error)

# Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(sample_train, label_train)

label_pred = clf.predict(sample_train)
j=0
for i in range(label_train.shape[0]):
    if label_train[i] == label_pred[i]:
        j = j+1

training_error = j/label_test.shape[0] * 100
print(training_error)


label_pred = clf.predict(sample_test)
j=0
for i in range(label_test.shape[0]):
    if label_test[i] == label_pred[i]:
        j = j+1

testing_error = j/label_test.shape[0] * 100
print(testing_error)


# Neural Networks
clf = MLPClassifier()
clf.fit(sample_train, label_train)

label_pred = clf.predict(sample_train)
j=0
for i in range(label_train.shape[0]):
    if label_train[i] == label_pred[i]:
        j = j+1

training_error = j/label_test.shape[0] * 100
print(training_error)


label_pred = clf.predict(sample_test)
j=0
for i in range(label_test.shape[0]):
    if label_test[i] == label_pred[i]:
        j = j+1

testing_error = j/label_test.shape[0] * 100
print(testing_error)
