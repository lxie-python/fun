# In order to find clusters within the data, we use some kind of similarity measure
# such as Euclidean distance. This similarity measure can estimate the tightness of a cluster.
# There is no universal similarity metric that works for all the cases. It depends on the problem at hand
# We might be interested in finding the representative data point for each subgroup
# or we might be interested in finding the outliers in our data. Depending on the situation,
# we will end up choosing the appropriate metric.
#######################################################################################
# K-Means: we start by fixing the number of clusters and classify our data based on that.
# The central idea here is that we need to update the locations of there K centroids with each iteration.
# We continue iteration until we have placed the centroids at their optimal locations.
# The initial placement of centroids plays an important role in the algorithm.
# There centroids should be placed in a clever manner.
# A good strategy is to place them as far away from each other as possible.
# The basic K-Means algorithm places there centroids randomly
# K-Means++ chooses these point algorithmically from the input list of data points.
# It tries to place the initial centroids far from each other so that it converges quickly.
# We then go through the training dataset and assign each data point to the closedt centroids.
# After the first iteration, we recalculate the location of the centroids based on the new clusters and repeat the process
# After a certain number of iterations, the centroids do not change their locations anymore

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

X=np.loadtxt('data_clustering.txt',delimiter=',')

num_clusters=5

# plt.figure()
# plt.scatter(X[:,0],X[:,1],marker='o',facecolor='none',edgecolors='black',s=80)
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
# plt.title('input data')
# plt.xlim(x_min,x_max)
# plt.ylim(y_min,y_max)
# plt.xticks(()) #not showing xticks
# plt.yticks(())
# plt.show()

#create KMeans object
kmeans=KMeans(init='k-means++',n_clusters=num_clusters,n_init=10)
kmeans.fit(X)
step_size=0.01

x_vals,y_vals=np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
output=kmeans.predict(np.c_[x_vals.ravel(),y_vals.ravel()])
output=output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output,interpolation='nearest',
           extent=(x_vals.min(),x_vals.max(),
                   y_vals.min(),y_vals.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto',origin='lower')

plt.scatter(X[:,0],X[:,1],marker='o',facecolor='none',edgecolors='black',s=80)
cluster_centers=kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker='o',s=210,linewidths=4,color='black',
            zorder=12,facecolor='black')

plt.title('k means')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()