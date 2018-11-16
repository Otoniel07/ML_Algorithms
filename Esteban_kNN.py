import csv
import random
import math
import operator
import numpy as np

data = np.loadtxt('CrimeCommunityBinary.csv', delimiter=',')
[n,p] = data.shape
data = np.c_[np.ones(n),data]

train_size = 30
trainingSet = data[0:train_size]
testSet = data[train_size:-1]

train_sample = data[0:train_size,0:-1]
train_label = data[0:train_size,-1]
test_sample = data[train_size:-1,0:-1]
test_label = data[train_size:-1,-1]
 
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
 
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getClass(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def knnError(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (1-(correct/float(len(testSet)))) * 100.0
	
def knn():
	# prepare data
    data = np.loadtxt('CrimeCommunityBinary.csv', delimiter=',')
    [n,p] = data.shape
    data = np.c_[np.ones(n),data]

    train_size = 50
    trainingSet = data[0:train_size]
    testSet = data[train_size:-1]
	# generate predictions
    predictions=[]
    k = 10
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getClass(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = knnError(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
	
knn()