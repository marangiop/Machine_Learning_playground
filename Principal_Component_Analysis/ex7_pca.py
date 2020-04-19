#!/usr/bin/env python

#  Principle Component Analysis on simple 2D dataset
#Corresponding to tasks listed on pages 9-12 from ex7.pdf document 

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import featureNormalize as fn
import pca
import drawLine as dl
import projectData as pd
import recoverData as rd
import displayData as dd

#import hsv 
#from mpl_toolkits.mplot3d import Axes3D

## ================== Part 1: Load Example Dataset  ===================
#  

print('Visualizing example dataset for PCA.\n');

mat = scipy.io.loadmat('ex7data1.mat')
X = np.array(mat["X"])



#  Visualize the example dataset
plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.\n');

#  Before running PCA, it is important to first normalize X
X_norm, mu, _ = fn.featureNormalize(X)

#  Run PCA
U, S = pca.pca(X_norm)
#U contains the principalcomponents; S will contains a diagonal matrix.

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.

dl.drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, c='k', linewidth=2)
dl.drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, c='k', linewidth=2)
plt.show()


print('Top eigenvector: \n')
print(' U(:,1) = {:f} {:f} \n'.format(U[0,0], U[1,0]))
#print('(you should expect to see -0.707107 -0.707107)')




## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.\n');

#  Plot the normalized dataset (returned from pca)
plt.scatter(X_norm[:,0], X_norm[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([-4, 3, -4, 3])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = pd.projectData(X_norm, U, K)

print('Projection of the first example: '+ str(Z[0]))

X_rec  = rd.recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
#print('(this value should be about  -1.047419 -1.047419)\n')


#  Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    dl.drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)

plt.show()
