# Logistic Regression

import numpy as np
import random
import time
import matplotlib.pyplot as plt

class LogisticRegression():

    def binaryFitByGradientDescent(self, xTrain, yTrain, theta = None , alpha = 0.0005, epsilon = 0.00000000000001, numOfIteration = 100000):
        startTime = time.process_time()
        self.initialVariables()
        self.xTrain = xTrain # variables
        self.yTrain = yTrain[:,np.newaxis] # target
        self.xTrans = np.hstack((np.ones((self.xTrain.shape[0],1)),self.xTrain))
        self.m = self.xTrans.shape[0]

        self.theta = np.zeros((self.xTrans.shape[1],1)) if theta is None else theta
        self.alpha = alpha
        self.epsilon = epsilon
        self.tempForCost = np.zeros_like(self.epsilon)
        self.numOfIteration = numOfIteration
        self.costJhistory = np.zeros(numOfIteration)

        for i in range(0,self.numOfIteration):
            # Improved Version -- Added Cost Function and Delta to determine the convergence level
            # Actually, compared with the Univariable LR, we can notice that the improved version of gradient descent algorithm
            # has the ability to deal with multiple variable LR.
            hypothesis = 1 / (1 + np.exp(- np.dot(self.xTrans,self.theta)))
            loss = hypothesis - self.yTrain
            cost = np.sum(- self.yTrain * np.log(hypothesis) - (1 - self.yTrain) * np.log(1 - hypothesis))
            self.costJhistory[i] = cost
            if (abs(cost - self.tempForCost)) <= self.epsilon :
                print('Cost changes Less than Delta %f \nIteration %d | Cost: %f' % (self.epsilon, i, cost))
                self.costJhistory = self.costJhistory[0:i+1] # slicing the variable self.costJhistory to only obtain the not-zero values
                break
            else:
                self.tempForCost = cost
            gradient = np.dot(self.xTrans.T,loss)/self.m
            self.theta = self.theta - self.alpha * gradient
            if (i == numOfIteration):
                print('The number of iteration achieved the %d \nIteration %d | Cost: %f' % (i, cost))

        print('\nCoefficients:')
        print(self.theta)
        endTime = time.process_time() - startTime
        print('\nProcessing Time: %f' % endTime)

        return self.theta

    def multipleFitByGradientDescent(self, xTrain, yTrain, theta = None , alpha = 0.0005, epsilon = 0.00000000000001, numOfIteration = 100000):
        self.yClass = np.unique(yTrain)
        self.thetas = np.zeros((xTrain.shape[1]+1, self.yClass.shape[0])) if theta is None else theta

        for i in range(self.yClass.shape[0]):
            yTrainTemp = np.array([1 if y == self.yClass[i] else 0 for y in yTrain])
            self.thetas[:,i] = self.binaryFitByGradientDescent(xTrain, yTrainTemp, alpha = alpha, epsilon = epsilon, numOfIteration = numOfIteration)[:,0]
            z = 1


    def initialVariables(self):
        self.xTrain = None
        self.yTrain = None
        self.xTrans = None
        self.m = None
        self.theta = None
        self.alpha = None
        self.epsilon = None
        self.tempForCost = None
        self.numOfIteration = None
        self.costJhistory = None

    def binaryPredict(self,xTest):

        self.xTest = xTest
        xTrans2 = np.hstack((np.ones((self.xTest.shape[0], 1)), self.xTest))
        self.yTest = 1 / (1 + np.exp(- np.dot(xTrans2,self.theta)))
        self.yTest = [0 if x < 0.5 else 1 for x in self.yTest]

        return  self.yTest

    def multiplePredict(self,xTest):

        self.xTest = xTest
        self.yResult = np.zeros((self.xTest.shape[0], self.yClass.shape[0]))
        xTrans2 = np.hstack((np.ones((self.xTest.shape[0], 1)), self.xTest))
        for i in range(self.yClass.shape[0]):
            self.yResult[:,i] = 1 / (1 + np.exp(- np.dot(xTrans2, self.thetas[:,i])))

        self.yResult = np.argmax(self.yResult,axis = 1)
        self.yTest = [self.yClass[i] for i in self.yResult]

        return  self.yTest





    @staticmethod
    def featureScaling(x):

        n,m = x.shape
        xScaled = np.ones_like(x)
        for j in range(m):
            max = np.max(x[:,j])
            min = np.min(x[:,j])
            avg = np.average(x[:,j])

            xScaled[:,j] = [(xij-avg)/(max-min) for xij in x[:,j]]

        return xScaled


# # Test by using real data (sklearn.datasets.load_diabetes())
# from sklearn import datasets
#
# iris = datasets.load_iris()
#
# x = iris.data[0:100]
# y = iris.target[0:100]
#
# # Random selection for the training set and testing set
# import random
# random.seed(1)
# indOfTest = random.sample(range(100),30)
# indOfTrain = [x for x in range(100) if x not in indOfTest]
# xTrain = x[indOfTrain]
# xTest = x[indOfTest]
# yTrain = y[indOfTrain]
# yTest = y[indOfTest]
# z = 1
#
# # Model training and predicting
# regr = LogisticRegression()
# theta = regr.binaryFitByGradientDescent(xTrain,yTrain,alpha=0.5,epsilon=0.00001,numOfIteration=500000)
# yResult = regr.binaryPredict(xTest)
#
# # Confusion matrix
# from sklearn.metrics import confusion_matrix
#
# conMatr = confusion_matrix(yResult,yTest)
# print(conMatr)
#
# # #plot the results
# # import matplotlib.pyplot as plt
# #
# # plt.scatter(xTrain,yTrain,color = 'black')
# # plt.plot(xTest,yTest,color='blue')
# # plt.show()






# Test by using real data (sklearn.datasets.load_diabetes())
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

# Random selection for the training set and testing set
import random
random.seed(1)
indOfTest = random.sample(range(150),50)
indOfTrain = [x for x in range(150) if x not in indOfTest]
xTrain = x[indOfTrain]
xTest = x[indOfTest]
yTrain = y[indOfTrain]
yTest = y[indOfTest]
z = 1

# Model training and predicting
regr = LogisticRegression()
theta = regr.multipleFitByGradientDescent(xTrain,yTrain,alpha=0.01,epsilon=0.00001,numOfIteration=500000)
yResult = regr.multiplePredict(xTest)

# Confusion matrix
from sklearn.metrics import confusion_matrix

conMatr = confusion_matrix(yResult,yTest)
print(conMatr)
