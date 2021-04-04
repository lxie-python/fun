import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

df = pd.read_excel(r'SNP.xls')
features=df.columns[1:-1]
label=df.columns[-1]

for i in features:   #nan数据都=mean
    if(df[i].isnull().sum())>0:
        pp=df[i].values.reshape(-1,1)
        imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')
        imp_mean=imp_mean.fit_transform(pp)
        df[i]=imp_mean

X=df[features]
# scale = MinMaxScaler().fit(df[features])   ## scale到range 0-1
# dataScale = scale.transform(df[features])

bandwidth_X=estimate_bandwidth(X,quantile=0.1)
meanshift_model=MeanShift(bandwidth=bandwidth_X,bin_seeding=True)
meanshift_model.fit(X)

cluster_centers=meanshift_model.cluster_centers_
print('\nCenters of clusters:',cluster_centers)

#estimate the number of clusters
labels=meanshift_model.labels_  #每一个数据所属于的cluster就是它的label
num_clusters=len(np.unique(labels))
print(num_clusters)

# plt.figure()
# markers='o*xvsd^v>'
# for i, marker in zip(range(num_clusters),markers):
#     plt.scatter(X[labels==i,0],X[labels==i,1],marker=marker,color='black')
#     cluster_center=cluster_centers[i]
#     plt.plot(cluster_center[0],cluster_center[1],marker='o',markersize=15,
#              markerfacecolor='black',markeredgecolor='black')
# plt.title('clusters')
# plt.show()