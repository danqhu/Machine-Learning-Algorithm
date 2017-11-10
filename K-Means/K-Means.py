import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

class KMeans():

    def fit(self, xTrain, numOfCluster, numOfIteration = 1000, epsilon = 0.00000001, numOfRandomInitialization = 10):

        self.initialization(xTrain, numOfCluster, numOfIteration, epsilon, numOfRandomInitialization)

        self.finalCostJHistory = list()
        self.finalCentroidsList = list()
        self.finalClosestCentroidsList = list()
        self.cluster = np.zeros_like(self.closestCentroidsList.shape)

        for numOfRandInit in range(self.numOfRandomInitialization):

            self.initializeCentroids(numOfRandInit)

            for i in range(self.numOfIteration):

                self.closestCentroidsList = self.findClosestCentroids(self.xTrain, self.centroids)
                newCentroids = self.updateCentroids(self.xTrain, self.closestCentroidsList)
                J = self.computeCostJ(self.xTrain,self.centroids,self.closestCentroidsList)
                self.costJHistory.append(J)

                if np.array_equal(self.centroids,newCentroids):
                    print('The centroids has been fixed and the number of Iteration is: %d | Cost: %f' %(i, J))
                    break
                elif J < self.epsilon:
                    print('The Cost has achieved epsilon %f and the number of Iteration is: %d | Cost: %f' % (self.epsilon,i, J))
                    break
                else:
                    self.centroids = newCentroids
                    if i == self.numOfIteration-1:
                        print('The number of Iteration has achieved the maximum %f | Cost: %f' % (self.numOfIteration, J))

            self.finalCostJHistory.append(self.costJHistory[-1])
            self.finalCentroidsList.append(self.centroids.copy())
            self.finalClosestCentroidsList.append(self.closestCentroidsList.copy())
            self.initialization(xTrain, numOfCluster, numOfIteration, epsilon, numOfRandomInitialization)

        a = np.argmin(self.finalCostJHistory)
        self.cluster = self.finalClosestCentroidsList[np.argmin(self.finalCostJHistory)]





    def initialization(self,xTrain, numOfCluster, numOfIteration, epsilon, numOfRandomInitialization):
        self.xTrain = xTrain
        self.m = self.xTrain.shape[0]
        self.K = numOfCluster
        self.numOfIteration = numOfIteration
        self.epsilon = epsilon
        self.numOfRandomInitialization = numOfRandomInitialization
        self.centroids = np.random.random((self.K, self.xTrain.shape[1]))
        self.costJHistory = list()
        self.closestCentroidsList = np.zeros((self.m, 1))

    def initializeCentroids(self, randomSeed):
        random.seed(randomSeed)
        indices = random.sample(range(self.m),self.K)
        self.centroids = self.xTrain[indices,:].copy()

    def findClosestCentroids(self, xTrain, centroids):
        m = xTrain.shape[0]
        closestCentroidsList = np.zeros((m,1),dtype=int)

        for i in range(m):

            closestCentroidsList[i,0] = np.argmin(np.sum((centroids - xTrain[i,:]) ** 2,axis = 1))

        return closestCentroidsList

    def updateCentroids(self, xTrain, closestCentroidsList):

        newCentroids = np.zeros_like(self.centroids)

        for i in range(self.K):
            xTrain_K = xTrain[np.where(closestCentroidsList == i)[0],:].copy()
            a = np.sum((xTrain_K)/xTrain_K.shape[0], axis=0)
            newCentroids[i,:] = np.sum((xTrain_K)/xTrain_K.shape[0], axis=0)
        return newCentroids

    def computeCostJ(self,xTrain, centroids, closestCentroidsList):
        distance = xTrain.copy()
        m = xTrain.shape[0]
        for i in range(m):
            a = closestCentroidsList[i,0]
            b = centroids[a,:]
            c = distance[i,:]
            distance[i,:] -= centroids[closestCentroidsList[i,0],:]

        J = np.sum(distance ** 2)/m

        return J







# Test by using real data (sklearn.datasets.load_diabetes())
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Random selection for the training set and testing set
import random
random.seed(1)

z = 1

# Model training and predicting
km = KMeans()
km.fit(x,3)
result = km.cluster

x_2Features = x[:,:2]
y_target = y
y_result = result

x_min, x_max = x_2Features[:, 0].min() - .5, x_2Features[:, 0].max() + .5
y_min, y_max = x_2Features[:, 1].min() - .5, x_2Features[:, 1].max() + .5

# plot the original class
plt.figure(1)
plt.clf()

plt.scatter(x_2Features[:, 0], x_2Features[:, 1], c=y_target, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("The original class")

# plot the clustered class

plt.figure(2)

plt.scatter(x_2Features[:, 0], x_2Features[:, 1], c=y_result, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("The clustered class")

plt.show()