# Linear-Regression-with-Multiple-Variables

import numpy as np
import random
import time

class LinearRegressionwithMultiplevariable():

    def fitByGradientDescent(self, xTrain, yTrain, theta = None , alpha = 0.0005, epsilon = 0.00000000000001, numOfIteration = 100000):
        startTime = time.process_time()
        self.xTrain = xTrain # variables
        self.yTrain = yTrain[:,np.newaxis] # target
        self.xTrans = np.hstack((np.ones((self.xTrain.shape[0],1)),self.xTrain))
        self.m = self.xTrans.shape[0]

        self.theta = np.zeros((self.xTrans.shape[1],1)) if theta is None else theta
        self.alpha = alpha
        self.epsilon = epsilon
        self.tempForCost = np.zeros_like(self.epsilon)
        self.numOfIteration = numOfIteration

        for i in range(0,self.numOfIteration):
            # Improved Version -- Added Cost Function and Delta to determine the convergence level
            # Actually, compared with the Univariable LR, we can notice that the improved version of gradient descent algorithm
            # has the ability to deal with multiple variable LR.
            hypothesis = np.dot(self.xTrans,self.theta)
            loss = hypothesis - self.yTrain
            cost = np.sum(loss ** 2)/(2*self.m)
            if (abs(cost - self.tempForCost)) <= self.epsilon :
                print('Cost changes Less than Delta %f \nIteration %d | Cost: %f' % (self.epsilon, i, cost))
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

    def fitByNormalEquation(self,xTrain, yTrain):
        startTime = time.process_time()
        self.xTrain = xTrain  # variables
        self.yTrain = yTrain[:, np.newaxis]  # target
        self.xTrans = np.hstack((np.ones((self.xTrain.shape[0], 1)), self.xTrain))

        self.theta = np.dot(np.dot(np.linalg.pinv(np.dot(self.xTrans.T,self.xTrans)),self.xTrans.T),self.yTrain)
        print('\nCoefficients:')
        print(self.theta)
        endTime = time.process_time() - startTime
        print('\nProcessing Time: %f' % endTime)

        return self.theta



    def predict(self,xTest):
        self.xTest = xTest
        xTrans2 = np.hstack((np.ones((self.xTest.shape[0], 1)), self.xTest))
        self.yTest = np.dot(xTrans2,self.theta)

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




# # Test by using simulated data
def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i,0] = i
        x[i,1] = i*2
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y
x, y = genData(100, 25, 10)

# Feature Scaling (After scaling, the coefficients obtained by the algorithm are different from the no scaled one, which we must use the scaled test set to test)
# x = LinearRegressionwithMultiplevariable.featureScaling(x)
xTrain = x
xTest = x
yTrain = y
yTest = y

# Model training and predicting
regr = LinearRegressionwithMultiplevariable()
# theta = regr.fitByGradientDescent(xTrain,yTrain,alpha=0.0001,epsilon=0.000001,numOfIteration=1000000)
theta = regr.fitByNormalEquation(xTrain,yTrain)
yTest = regr.predict(xTest)

#plot the results
import matplotlib.pyplot as plt

plt.scatter(xTrain[:,0],yTrain,color = 'black')
plt.plot(xTest[:,0],yTest,color='blue')
plt.show()



# # Test by using real data (sklearn.datasets.load_diabetes())
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
# theta = regr.fit(xTrain,yTrain,alpha=0.5,delta=0.00001,numOfIteration=500000)
# yTest = regr.predict(xTest)
#
# #plot the results
# import matplotlib.pyplot as plt
#
# plt.scatter(xTrain,yTrain,color = 'black')
# plt.plot(xTest,yTest,color='blue')
# plt.show()



# # Test by comparing with the scikit-learn package: linear_model.LinearRegression()
# from sklearn import datasets, linear_model
#
# diabetes = datasets.load_diabetes()
#
# # Add a dimension for the array
# x = diabetes.data[:,np.newaxis,2]
# y = diabetes.target
#
# xTrain = x[:-20]
# xTest = x[-20:]
# yTrain = y[:-20]
# yTest = y[-20:]
#
# regr = linear_model.LinearRegression()
# regr.fit(xTrain,yTrain)
# yTest = regr.predict(xTest)
#
#
# #plot the results
# import matplotlib.pyplot as plt
#
# plt.scatter(xTrain,yTrain,color = 'black')
# plt.plot(xTest,yTest,color='blue')
# plt.show()
