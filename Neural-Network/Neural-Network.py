
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

class NeuralNetwork():

    def fitByGradientDescent(self, xTrain, yTrain, num_labels, hidden_layer_size,alpha = 0.01, epsilon = 0.0001, numOfIteration = 100, input_layer_size = None, Lambda = 1,IsCheckGradient = False):
        startTime = time.process_time()
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTrans = np.hstack((np.ones((self.xTrain.shape[0],1)),self.xTrain))

        self.m = xTrain.shape[0]
        self.numOfIteration = numOfIteration
        self.input_layer_size = input_layer_size if input_layer_size is not None else self.xTrain.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.num_labels = num_labels
        self.alpha = alpha
        self.epsilon = epsilon
        self.Lambda = Lambda
        self.netWorkArchitecture = [self.input_layer_size] + self.hidden_layer_size + [self.num_labels]
        self.costJHistory = list()
        self.Thetas = list()

        # initialize Thetas
        for i in range(len(self.netWorkArchitecture) - 1):
            numOut = self.netWorkArchitecture[i + 1]
            numIn = self.netWorkArchitecture[i]
            ThetaTemp = self.randInitialWeights(numIn,numOut,epsilon=0.12)
            self.Thetas.append(ThetaTemp)

        self.yBinarilization()

        OldJ = 0
        for i in range(self.numOfIteration):

            J, ThetasGrad = self.nnCostFunction()

            if IsCheckGradient is True:
                checkResult = self.checkNNGradient();
                diff = np.linalg.norm(checkResult[0]-ThetasGrad[0],2)/np.linalg.norm(checkResult[0]+ThetasGrad[0],2)

            self.costJHistory.append(J)

            if abs(OldJ - J) < self.epsilon and (OldJ - J) > 0:
                print('Cost changes Less than Epsilon %f \nIteration %d | Cost: %f' % (self.epsilon, i, J))
                break
            elif (OldJ - J) < 0 and i != 0:
                print('Cost begins increasing from %f to %f \nIteration %d | Cost: %f' % (OldJ, J, i, J))
                break
            else:
                OldJ = J
                for j in range(len(self.Thetas)):
                    self.Thetas[j] = self.Thetas[j] - self.alpha * ThetasGrad[j]

            if (i == numOfIteration):
                print('The number of iteration achieved the %d \nIteration %d | Cost: %f' % (i, j))
            else:

                print('The number of iteration is %d | Cost: %f' % (i, J))

        endTime = time.process_time() - startTime
        print('\nProcessing Time: %f' % endTime)


    def nnCostFunction(self):

        # Cost function calculation
        zListCost = list()
        aListCost = list()
        currentIn = self.xTrans
        for i in range(len(self.Thetas)):
            zTemp = np.dot(currentIn, self.Thetas[i].T)
            aTemp = self.sigmoid(zTemp)
            if i != len(self.Thetas)-1:
                aTemp = np.hstack((np.ones((aTemp.shape[0],1)), aTemp))
            zListCost.append(zTemp)
            aListCost.append(aTemp)
            currentIn = aTemp

        J = 1/self.m * np.sum(np.sum(- self.yTrainBinary * np.log(aListCost[-1]) - (1 - self.yTrainBinary) * np.log(1 - aListCost[-1]),axis=1)) \
            + self.Lambda/(2*self.m) * np.sum([np.sum(theta[:,1:] ** 2) for theta in self.Thetas])

        # Gradient for thetas
        # Initialize the ThetasGrad
        ThetasGrad = list()
        for i in range(len(self.Thetas)):
            ThetasGrad.append(np.zeros((self.Thetas[i].shape)))

        for i in range(self.m):

            currentIn = self.xTrans[i,:][np.newaxis,:].T
            aList, zList = self.forwardPropagation(currentIn)


            Deltas = self.backPropagation(aList, zList, i)

            ThetasGrad[0] += np.dot(Deltas[0], self.xTrans[i, :][np.newaxis,:])
            for i in range(1, len(self.Thetas)):
                ThetasGrad[i] += np.dot(Deltas[i], aList[i-1].T)

        # calculate the regularization items and add to the ThetasGrad
        ThetasReg = list()
        for i in range(len(self.Thetas)):
            ThetasReg.append(self.Thetas[i].copy())

        for i in range(len(ThetasReg)):
            ThetasReg[i][:,0] = 0
            ThetasGrad[i] = ThetasGrad[i]/self.m - self.Lambda/self.m * ThetasReg[i]

        return J,ThetasGrad

    def sigmoid(self, z):
        return 1 / (1 + np.exp(- z))

    def sigmoidGradient(self,z):
        return self.sigmoid(z) * (1- self.sigmoid(z))

    def forwardPropagation(self, currentIn):
        zListGrad = list()
        aListGrad = list()
        currentInTemp = currentIn
        for i in range(len(self.Thetas)):
            zTemp = np.dot(self.Thetas[i], currentInTemp)
            aTemp = self.sigmoid(zTemp)
            if i != len(self.Thetas)-1:
                aTemp = np.vstack(([1], aTemp))
            zListGrad.append(zTemp)
            aListGrad.append(aTemp)
            currentInTemp = aTemp

        return aListGrad,zListGrad

    def backPropagation(self, aList, zList, indexOfY):
        Deltas = list()
        deltaOutput = aList[-1] - self.yTrainBinary[indexOfY,:][np.newaxis,:].T
        Deltas.insert(0,deltaOutput)
        for i in range(len(self.Thetas)-1,0,-1):
            deltaTemp = np.dot(self.Thetas[i].T, Deltas[0])[1:,:] * self.sigmoidGradient(zList[i-1])
            Deltas.insert(0,deltaTemp)
        return Deltas

    def yBinarilization(self):
        # the items in the binarilized yTrain
        self.yClass = np.unique(self.yTrain)[np.newaxis,:]
        self.yTrainBinary = np.zeros((self.m, self.num_labels))
        for i in range(self.m):
            self.yTrainBinary[i,np.where(self.yClass == self.yTrain[i])[1][0]] = 1


    def randInitialWeights(self, numIn, numOut, epsilon = None):

        epsilon = epsilon if epsilon is not None else np.sqrt(6)/np.sqrt(numIn + numOut)
        W = np.random.rand(numOut, numIn + 1) * 2 * epsilon - epsilon

        return W

    def checkNNGradient(self):

        ThetasCheck = self.copy(self.Thetas)

        ThetasGrad = self.copy(self.Thetas)
        epsilon = 0.0001

        for i in range(len(ThetasCheck)):
            for j in range(ThetasCheck[i].shape[0]):
                for l in range(ThetasCheck[i].shape[1]):
                    ThetaUp = self.copy(self.Thetas)
                    ThetaUp[i][j,l] += epsilon
                    ThetaDown = self.copy(self.Thetas)
                    ThetaDown[i][j, l] -= epsilon
                    ThetasGrad[i][j,l] = (self.costJCalculation(ThetaUp) - self.costJCalculation(ThetaDown))/(2 * epsilon)
        return ThetasGrad

    def copy(self,object):
        ThetaTemp = list()
        for i in range(len(object)):
            ThetaTemp.append(object[i].copy())
        return ThetaTemp



    def costJCalculation(self, ThetaCheck):
        # Cost function calculation
        zListCost = list()
        aListCost = list()
        currentIn = self.xTrans
        for i in range(len(ThetaCheck)):
            zTemp = np.dot(currentIn, ThetaCheck[i].T)
            aTemp = self.sigmoid(zTemp)
            if i != len(ThetaCheck) - 1:
                aTemp = np.hstack((np.ones((aTemp.shape[0], 1)), aTemp))
            zListCost.append(zTemp)
            aListCost.append(aTemp)
            currentIn = aTemp

        J = 1 / self.m * np.sum(
            np.sum(- self.yTrainBinary * np.log(aListCost[-1]) - (1 - self.yTrainBinary) * np.log(1 - aListCost[-1]),
                   axis=1)) \
            + self.Lambda / (2 * self.m) * np.sum([np.sum(theta[:, 1:] ** 2) for theta in ThetaCheck])
        return J

    # def debugInitializeWeights(self, numIn, numOut):
    #     W = np.zeros(numOut, numIn +1)
    #     W = np.array([np.sin(x) for x in range(1,W.size)]).reshape((W.shape))
    #
    #     return W


    def predict(self, xTest):

        self.xTest = xTest
        self.yResult = np.zeros((self.xTest.shape[0], self.yClass.shape[1]))
        xTrans2 = np.hstack((np.ones((self.xTest.shape[0], 1)), self.xTest))
        for i in range(self.xTest.shape[0]):
            aList, zList = self.forwardPropagation(xTrans2[i,:][np.newaxis,:].T)
            self.yResult[i,:] = aList[-1].T

        self.yResult = np.argmax(self.yResult, axis=1)
        self.yTest = [self.yClass[0,i] for i in self.yResult]

        return self.yTest


#
# # Test by using real data (sklearn.datasets.load_diabetes())
# from sklearn import datasets
#
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
#
# # Random selection for the training set and testing set
# import random
# random.seed(1)
# indOfTest = random.sample(range(150),50)
# indOfTrain = [x for x in range(150) if x not in indOfTest]
# xTrain = x[indOfTrain]
# xTest = x[indOfTest]
# yTrain = y[indOfTrain]
# yTest = y[indOfTest]
# z = 1
#
# # Model training and predicting
# nn = NeuralNetwork()
# nn.fitByGradientDescent(xTrain,yTrain,num_labels=3,hidden_layer_size = [5],alpha=0.1,epsilon=0.00000005,numOfIteration=10000,Lambda=0.1,IsCheckGradient= False)
# results = nn.predict(xTest)
#
# # Confusion matrix
# from sklearn.metrics import confusion_matrix
#
# conMatr = confusion_matrix(results,yTest)
# print(conMatr)


# handwriting recognition test

import scipy.io as sio

dataset = sio.loadmat('ex4data1.mat')

x = dataset['X']
y = dataset['y']

# Random selection for the training set and testing set
import random
random.seed(1)
indOfTest = random.sample(range(5000),1000)
indOfTrain = [x for x in range(5000) if x not in indOfTest]
xTrain = x[indOfTrain]
xTest = x[indOfTest]
yTrain = y[indOfTrain]
yTest = y[indOfTest]

# Model training and predicting
nnforHandWriting = NeuralNetwork()
nnforHandWriting.fitByGradientDescent(xTrain,yTrain,num_labels=10,hidden_layer_size = [25],alpha=0.5,epsilon=0.00000001,numOfIteration=10000,Lambda=1,IsCheckGradient= False)
results = nnforHandWriting.predict(xTest)

# Confusion matrix
from sklearn.metrics import confusion_matrix

conMatr = confusion_matrix(results,yTest)
print(conMatr)

