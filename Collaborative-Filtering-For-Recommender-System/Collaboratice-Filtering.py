import numpy as np
import time
import matplotlib.pyplot as plt

class CollaborativeFiltering():

    def fit(self, Y, R, numOfFeatures = 10, alpha = 0.1, Lambda = 0, epsilon = 0.001, numOfIteration = 100):
        startTime = time.process_time()
        self.Y = Y
        self.R = R
        self.numOfFeatures = numOfFeatures
        self.alpha = alpha
        self.Lambda = Lambda
        self.epsilon = epsilon
        self.numOfIteration = numOfIteration
        self.numOfUsers = Y.shape[1]
        self.numOfMovies = Y.shape[0]
        self.X = np.random.rand(self.numOfMovies, self.numOfFeatures)
        self.Theta = np.random.rand(self.numOfUsers, self.numOfFeatures)
        self.costJHistory = list()

        self.normalize()

        OldJ = 0
        for i in range(self.numOfIteration):

            J, X_grad, Theta_grad = self.cofiCostFunction(self.X,self.Theta,self.Y,self.R,self.Lambda,self.numOfUsers,self.numOfMovies)

            if abs(OldJ - J) < self.epsilon and (OldJ - J) > 0:
                print('Cost changes Less than Epsilon %f \nIteration %d | Cost: %f' % (self.epsilon, i, J))
                break
            elif (OldJ - J) < 0 and i != 0:
                print('Cost begins increasing from %f to %f \nIteration %d | Cost: %f' % (OldJ, J, i, J))
                break
            else:
                self.costJHistory.append(J)
                OldJ = J
                self.Theta = self.Theta - self.alpha * Theta_grad
                self.X = self.X - self.alpha * X_grad


            if (i == numOfIteration):
                print('The number of iteration achieved the %d \nIteration %d | Cost: %f' % (i, J))
            else:

                print('The number of iteration is %d | Cost: %f' % (i, J))

        endTime = time.process_time() - startTime
        print('\nProcessing Time: %f' % endTime)

    def normalize(self):
        m,n = self.Y.shape
        self.YMean = np.zeros((m,1))
        self.YNorm = np.zeros_like(self.Y)

        for i in range(m):
            idx = np.where(self.R[i,:]==1)[0].tolist()
            # temp = self.Y[i, idx]
            self.YMean[i,0] = np.mean(self.Y[i, idx])
            self.YNorm[i,idx] = self.Y[i,idx] - self.YMean[i,0]

        self.Y = self.YNorm

    def cofiCostFunction(self, X, Theta, Y, R, Lambda, numOfUsers, numOfMovies):


        X_grad = np.zeros_like(X)
        Theta_grad = np.zeros_like(Theta)

        J = np.sum(np.sum((np.dot(X, Theta.T) - Y) ** 2 * R, axis = 1))/2 + Lambda/2 * np.sum(np.sum(Theta ** 2,axis =1))\
            + Lambda/2 * np.sum(np.sum(X ** 2,axis =1))

        for i in range(numOfMovies):
            idx = np.where(R[i,:]==1)[0].tolist()
            ThetaTemp = Theta[idx,:].copy()
            YTemp = Y[i,idx].copy()
            X_grad[i,:] = np.dot((np.dot(X[i,:], ThetaTemp.T) - YTemp), ThetaTemp) + Lambda * X[i,:]

        for j in range(numOfUsers):
            idx = np.where(R[:,j]==1)[0].tolist()
            XTemp = X[idx,:].copy()
            YTemp = Y[idx,j].copy()
            Theta_grad[j,:] = np.dot((np.dot(XTemp, Theta[j,:].T) - YTemp).T, XTemp) + Lambda * Theta[j,:]

        return J, X_grad, Theta_grad





# Test using dataset from the coursera machine learning exercise 8

import scipy.io as sio

dataset = sio.loadmat('ex8_movies.mat')
parameters = sio.loadmat('ex8_movieParams.mat')


Y = dataset['Y']
R = dataset['R']
X = parameters['X']
Theta = parameters['Theta']

# # using subset to test the algorithm whether working correctly
# numOfUsers = parameters['num_users']
# numOfMovies = parameters['num_movies']
# numOfFeatures = parameters['num_features']
#
# numOfUsers = 4
# numOfMovies = 5
# numOfFeatures = 3
#
# YTrain = Y[:numOfMovies,:numOfUsers]
# RTrain = R[:numOfMovies,:numOfUsers]
#
# XTrain = X[:numOfMovies,:numOfFeatures]
# ThetaTrain = Theta[:numOfUsers,:numOfFeatures]
#
# cofi = CollaborativeFiltering()
# J,X_grad,Theta_gras = cofi.cofiCostFunction(XTrain,ThetaTrain,YTrain,RTrain,1.5,numOfUsers,numOfMovies)
# i = 1



myRatings = np.zeros((Y.shape[0],1))
myRatings[0] = 4;

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
myRatings[97] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
myRatings[6] = 3;
myRatings[11]= 5;
myRatings[53] = 4;
myRatings[63]= 5;
myRatings[65]= 3;
myRatings[68] = 5;
myRatings[182] = 4;
myRatings[225] = 5;
myRatings[354]= 5;

Y = np.hstack((myRatings, Y))

myR = np.zeros((R.shape[0],1))
myR[np.where(myRatings != 0)[0].tolist(),0]=1

R = np.hstack((myR, R))



cofi = CollaborativeFiltering()
cofi.fit(Y, R,numOfIteration=2000,alpha=0.002,Lambda=10,epsilon=0.05)
Theta = cofi.Theta
X = cofi.X


movieList = {}
with open('movie_ids_utf_8.txt','r') as f:
    for line in f:
        movieList[line.split(' ')[0]] = line.split(' ',maxsplit = 1)[1].strip()

p = np.dot(X, Theta.T)

myRecommendation = p[:,0][:,np.newaxis] + cofi.YMean

myRecomIdx = np.argsort(myRecommendation.reshape((myRecommendation.shape[0])))

for i in range(-1,-11,-1):
    j = myRecomIdx[i]
    print('Predicting rating %f for movie %s\n'%(myRecommendation[j,0],movieList['%s'%(j+1)]))






