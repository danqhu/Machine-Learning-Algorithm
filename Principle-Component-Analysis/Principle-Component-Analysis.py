import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

class PCA():

    def pca(self, xTrain, IsFeatureNormalization = True):
        self.xTrain = xTrain
        self.m, self.n = self.xTrain.shape

        #feature normalization
        if IsFeatureNormalization is True:

            self.mu = np.mean(self.xTrain,axis=0)
            self.xTrain_norm = self.xTrain - self.mu

            self.sigma = np.std(self.xTrain_norm, axis = 0, ddof = 1)

            self.xTrain_norm = self.xTrain_norm/self.sigma

            self.xTrain = self.xTrain_norm








        self.Sigma = np.dot(self.xTrain.T, self.xTrain)/self.m
        self.U, self.S, self.V = np.linalg.svd(self.Sigma)

    def projectData(self, K):
        self.K = K

        self.Ureduce = self.U[:,:self.K]
        self.Z = np.dot(self.xTrain, self.Ureduce)

        return self.Z

    def recoverData(self):
        self.xTrain_rec = np.dot(self.Z, self.Ureduce.T)

        return self.xTrain_rec

    def projectDataWithoutK(self,epsilon):
        self.epsilon = epsilon

        for i in range(1,self.n+1):
            varianceLoss = 1 - np.sum(self.S[:i])/np.sum(self.S)
            if varianceLoss < self.epsilon:
                self.K = i
                break

        return self.projectData(self.K)













# Test by using real data (sklearn.datasets.load_diabetes())
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target


# Model training and predicting
pca_3 = PCA()
pca_3.pca(x)
xTrain_proj_3 = pca_3.projectData(3)
xTrain_rec_3 = pca_3.recoverData()

pca_2 = PCA()
pca_2.pca(x)
xTrain_proj_2 = pca_2.projectData(2)
xTrain_rec_2 = pca_2.recoverData()

pca_unknow = PCA()
pca_unknow.pca(x)
xTrain_proj_unknow = pca_unknow.projectDataWithoutK(0.0000001)
xTrain_rec_unknow = pca_unknow.recoverData()

# import some data to play with
X = iris.data[:, :2]  # we only take the first two features.

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5






# To getter a better understanding of interaction of the dimensions
# plot the First three PCA directions of original data
fig = plt.figure(1,figsize=(8, 6))
plt.clf()

ax1 = Axes3D(fig, elev = -150, azim=110)
ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax1.set_title("First three PCA directions of original data")
ax1.set_xlabel("1st eigenvector")
ax1.w_xaxis.set_ticklabels([])
ax1.set_ylabel("2nd eigenvector")
ax1.w_yaxis.set_ticklabels([])
ax1.set_zlabel("3rd eigenvector")
ax1.w_zaxis.set_ticklabels([])


# plot the First three PCA directions calculated by sklearn
fig = plt.figure(2,figsize=(8, 6))
ax2 = Axes3D(fig, elev=-150, azim=110)
X_reduced_3 = decomposition.PCA(n_components=3).fit_transform(iris.data)
ax2.scatter(X_reduced_3[:, 0], X_reduced_3[:, 1], X_reduced_3[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax2.set_title("First three PCA directions calculated by sklearn")
ax2.set_xlabel("1st eigenvector")
ax2.w_xaxis.set_ticklabels([])
ax2.set_ylabel("2nd eigenvector")
ax2.w_yaxis.set_ticklabels([])
ax2.set_zlabel("3rd eigenvector")
ax2.w_zaxis.set_ticklabels([])


# plot the First three PCA directions calculated by PCA
fig = plt.figure(3,figsize=(8, 6))
ax3 = Axes3D(fig, elev=-150, azim=110)

ax3.scatter(xTrain_proj_3[:, 0], xTrain_proj_3[:, 1], xTrain_proj_3[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax3.set_title("First three PCA directions calculated by PCA")
ax3.set_xlabel("1st eigenvector")
ax3.w_xaxis.set_ticklabels([])
ax3.set_ylabel("2nd eigenvector")
ax3.w_yaxis.set_ticklabels([])
ax3.set_zlabel("3rd eigenvector")
ax3.w_zaxis.set_ticklabels([])


# plot the original data
plt.figure(4)
plt.subplot(131)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("The original data")


# plot the original data
plt.subplot(132)
X_reduced_2 = decomposition.PCA(n_components=2).fit_transform(iris.data)
plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("The data reduced by sklearn")


# plot the original data
plt.subplot(133)

plt.scatter(xTrain_proj_2[:, 0], xTrain_proj_2[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("The data reduced by PCA")



plt.show()
