import numpy as np
import matplotlib.pyplot as plt

class GaussianDistribution():

    def estimateGaussuan(self, xTrain, epsilon = 0.5):

        self.xTrain = xTrain
        self.m, self.n = self.xTrain.shape
        self.epsilon = epsilon
        self.mu = np.mean(xTrain, axis=0)[np.newaxis,:]
        self.sigma2 = np.var(xTrain, axis = 0)[np.newaxis,:]

    def probabilityCalculation(self,xTest, IsMultiVariateGaussian = False):

        self.IsMultiVariateGaussian = IsMultiVariateGaussian
        # false solution(abandoned)
        # self.Sigma2 = np.dot(self.xTrain.T, self.xTrain)/self.m
        self.Sigma2 = np.cov(self.xTrain, rowvar=False)
        if IsMultiVariateGaussian is False:
            self.Sigma2 = self.Sigma2 * np.eye(self.Sigma2.shape[0])

        k = self.mu.shape[1]
        X = xTest - self.mu
        p = (2 * np.pi) ** (-k / 2) * np.linalg.det(self.Sigma2) ** (-0.5) * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(self.Sigma2)) * X,axis=1))

        return p

    def selectThreshold(self,yVal, pVal):

        self.bestEpsilon = 0
        self.bestF1 = 0
        F1 = 0

        stepsize = (max(pVal) - min(pVal))/1000
        epsilon = min(pVal)
        while epsilon <= max(pVal):
            predictions = [ 1 if p < epsilon else 0 for p in pVal]
            tp = sum([1 if predictions[i] ==1 and yVal[i] == 1 else 0 for i in range(len(predictions))])
            fp = sum([1 if predictions[i] == 1 and yVal[i] == 0 else 0 for i in range(len(predictions))])
            fn = sum([1 if predictions[i] == 0 and yVal[i] == 1 else 0 for i in range(len(predictions))])

            if tp == 0:
                prec = 0
                rec = 0
                F1 = 0
            else:
                prec = tp/(tp + fp)
                rec = tp/(tp + fn)
                F1 = 2 * prec * rec /(prec + rec)

            if F1 > self.bestF1:
                self.bestF1 = F1
                self.bestEpsilon = epsilon

            epsilon += stepsize

        # utilizing the p in pVal to determine the threshold of the model(the same as ROC curve)
        # for epsilon in np.unique(pVal):
        #     predictions = [ 1 if p < epsilon else 0 for p in pVal]
        #     tp = sum([1 if predictions[i] ==1 and yVal[i] == 1 else 0 for i in range(len(predictions))])
        #     fp = sum([1 if predictions[i] == 1 and yVal[i] == 0 else 0 for i in range(len(predictions))])
        #     fn = sum([1 if predictions[i] == 0 and yVal[i] == 1 else 0 for i in range(len(predictions))])
        #
        #     if tp == 0:
        #         prec = 0
        #         rec = 0
        #         F1 = 0
        #     else:
        #         prec = tp/(tp + fp)
        #         rec = tp/(tp + fn)
        #         F1 = 2 * prec * rec /(prec + rec)
        #
        #     if F1 > self.bestF1:
        #         self.bestF1 = F1
        #         self.bestEpsilon = epsilon

        return self.bestEpsilon, self.bestF1





# Test using dataset from the coursera machine learning exercise 8

import scipy.io as sio

dataset = sio.loadmat('ex8data1.mat')

x = dataset['X']
xVal = dataset['Xval']
yVal = dataset['yval']


gd = GaussianDistribution()
gd.estimateGaussuan(x)
result = gd.probabilityCalculation(x, IsMultiVariateGaussian=True)
pVal = gd.probabilityCalculation(xVal, IsMultiVariateGaussian=True)
epsilon, F1 = gd.selectThreshold(yVal,pVal)

print('(you should see a value epsilon of about 8.99e-05)\n');
print('The best epsilon for xVal is %g\n' % (epsilon))
print('\n(you should see a Best F1 value of  0.875000)\n');
print('The Best F1 for xVal is %f\n' % (F1))

dataset2 = sio.loadmat('ex8data2.mat')

x = dataset2['X']
xVal = dataset2['Xval']
yVal = dataset2['yval']

gd = GaussianDistribution()
gd.estimateGaussuan(x)

# if we use the multivariateGaussian method to estimate the parameters for the model,
# the final result will degenerate
result = gd.probabilityCalculation(x, IsMultiVariateGaussian=False)
pVal = gd.probabilityCalculation(xVal, IsMultiVariateGaussian=False)
epsilon, F1 = gd.selectThreshold(yVal,pVal)

print('(you should see a value epsilon of about 1.38e-18)\n');
print('The best epsilon for xVal is %g\n' % (epsilon))
print('\n(you should see a Best F1 value of  0.615385)\n');
print('The Best F1 for xVal is %f\n' % (F1))