# Linear-Regression-with-One-Variable
# DataSet: sklearn.datasets.load_diabetes()
#


import numpy as np
import random


class LinearRegressionwithUnivariable():

    def fit(self, xTrain, yTrain, theta = np.zeros(2), alpha = 0.0005, delta = 0.00000000000001, numOfIteration = 100000):
        self.xTrain = xTrain # uinvariable
        self.yTrain = yTrain # target
        self.xTrans = np.vstack((np.ones_like(self.xTrain),self.xTrain)).T
        self.m = self.xTrans.shape[0]
        self.theta = theta
        self.tempForTheta = np.zeros_like(self.theta)
        self.alpha = alpha
        self.delta = delta
        self.tempForCost = np.zeros_like(self.delta)
        self.numOfIteration = numOfIteration

        for i in range(0,self.numOfIteration):
            # # Original Version based the equations in the Machine Learning course on Coursera
            # self.tempForTheta[0] = self.theta[0] - self.alpha/self.m*(np.sum(np.dot(self.xTrans,self.theta)-self.yTrain))
            # self.tempForTheta[1]= self.theta[1] - self.alpha/self.m*(np.sum((np.dot(self.xTrans,self.theta)-self.yTrain)*self.xTrain))
            #
            # self.theta = self.tempForTheta

            # Improved Version -- Added Cost Function and Delta to determine the convergence level
            hypothesis = np.dot(self.xTrans,self.theta)
            loss = hypothesis - self.yTrain
            cost = np.sum(loss ** 2)/(2*self.m)
            if (abs(cost - self.tempForCost)) <= self.delta :
                print('Cost changes Less than Delta %f \nIteration %d | Cost: %f' % (self.delta, i, cost))
                break
            else:
                self.tempForCost = cost
            gradient = np.dot(self.xTrans.T,loss)/self.m
            self.theta = self.theta - self.alpha * gradient
            if (i == numOfIteration):
                print('The number of iteration achieved the %d \nIteration %d | Cost: %f' % (i, cost))

        print('\nCoefficients:\nTheta_0: %f    Theta_1: %f' % (self.theta[0], self.theta[1]))

        return self.theta

    def predict(self,xTest):
        self.xTest = xTest
        xTrans2 = np.vstack((np.ones_like(self.xTest),self.xTest)).T
        self.yTest = np.dot(xTrans2,self.theta)

        return  self.yTest



# # Test by using simulated data
def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y
x, y = genData(100, 25, 10)

xTrain = x
xTest = x
yTrain = y
yTest = y

# Model training and predicting
regr = LinearRegressionwithUnivariable()
theta = regr.fit(xTrain,yTrain,alpha=0.0005,delta=0.00000001,numOfIteration=100000)
yTest = regr.predict(xTest)

#plot the results
import matplotlib.pyplot as plt

plt.scatter(xTrain,yTrain,color = 'black')
plt.plot(xTest,yTest,color='blue')
plt.show()



# Test by using real data (sklearn.datasets.load_diabetes())
# from sklearn import datasets
#
# diabetes = datasets.load_diabetes()
#
# x = diabetes.data[:,2]
# y = diabetes.target
# xTrain = x[:-20]
# xTest = x[-20:]
# yTrain = y[:-20]
# yTest = y[-20:]
#
# # Model training and predicting
# regr = LinearRegressionwithUnivariable()
# theta = regr.fit(xTrain,yTrain,alpha=0.001,delta=0.001,numOfIteration=500000)
# yTest = regr.predict(xTest)
#
# #plot the results
# import matplotlib.pyplot as plt
#
# plt.scatter(xTrain,yTrain,color = 'black')
# plt.plot(xTest,yTest,color='blue')
# plt.show()
