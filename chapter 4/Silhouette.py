# Silhouette gives an estimate of how well each data point fits with its cluster.
# The silhouette score is a metric that measures how similar a data point is to its own cluster,
# as compared to other clusters.
# silhouette score=(p-q)/max(p,q) p是mean distanc到旁边cluster，q是mean distance到own cluster
# score从[-1,1]，越接近1越好，越接近-1越表示这个point与cluster其他的点越不像
# if you get too many points with negative silhouette scores, then we may have too few or
# too many clusters in our data. We need to run the clustering algorithm again to find
# the optimal number of clusters

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

X=np.loadtxt('data_quality.txt',delimiter=',')
scores=[]
values=np.arange(2,10)

for num_clusters in values:
    kmeans=KMeans(init='k-means++',n_clusters=num_clusters,n_init=10)
    kmeans.fit(X)
    score=metrics.silhouette_score(X,kmeans.labels_,metric='euclidean',sample_size=len(X))
    print('\nNumber of clusters=',num_clusters)
    print('Silhouette score=',score)
    scores.append(score)

plt.figure()
plt.bar(values,scores,width=0.7,color='black',align='center')
plt.title('Silhouette score vs number of clusters')
plt.show()

num_clusters=np.argmax(scores)+values[0]
print('\nOptimal number of clusters=',num_clusters)

