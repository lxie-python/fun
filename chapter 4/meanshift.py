# Mean shift is a non-parametric algorithm because it dose not make any assumptions about the
# underlying distributions.
# Mean Shift算法是一种无参密度估计算法或称核密度估计算法，Mean shift是一个向量，它的方向指向当前点上概率密度梯度的方向。
# In the mean shift algorithm, we consider the whole feature space as a probability density function.
# We start with the training dataset and assume that they have been sampled from a
# probability density function.
# In this framework, the clusters correspond to the local maxima of the underlying distribution.
# If there are K clusters, then there are K peaks in the underlying data distribution
# and Mean Shift will identify those peaks.
# For each data point in the training dataset, it defines a window around it.
# It then computes the centroid for this window and updates the location to this new centroid.
# It then repeats the process for this new location by defining a window around it.
# As we keep doing this, we move closer to the peak of the cluster.
# Each data point will move towards the cluster it belongs to.
# The movement is towards a region of higher density.
# We keep shifting the centroids, also called means, towards the peaks of each cluster--Mean Shift
# We keep doing this until the algorithm converges, at which stage the centroids don't move anymore.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift,estimate_bandwidth
from itertools import cycle

X=np.loadtxt('data_clustering.txt',delimiter=',')
#先预估数据带宽，如果MeanShift函数没有传入bandwidth参数，MeanShift会自动运行estimate_bandwidth
#quantile的值表示进行近邻搜索时近邻占样本的比例
bandwidth_X=estimate_bandwidth(X,quantile=0.1,n_samples=len(X))
meanshift_model=MeanShift(bandwidth=bandwidth_X,bin_seeding=True)
meanshift_model.fit(X)

cluster_centers=meanshift_model.cluster_centers_
print('\nCenters of clusters:',cluster_centers)

#estimate the number of clusters
labels=meanshift_model.labels_  #每一个数据所属于的cluster就是它的label
num_clusters=len(np.unique(labels))

plt.figure()
markers='o*xvs'
colors=['blue','green','pink','black','yellow']
for i, marker, color in zip(range(num_clusters),markers,colors):
    plt.scatter(X[labels==i,0],X[labels==i,1],marker=marker,color=color)
    cluster_center=cluster_centers[i]
    plt.plot(cluster_center[0],cluster_center[1],marker='o',markersize=15,
             markerfacecolor='black',markeredgecolor='black')
plt.title('clusters')
plt.show()