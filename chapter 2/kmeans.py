from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import xlrd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_excel(r'suoyou.xls')
features=df.columns[1:] #features
target=df.columns[0]  #label
y=df[target]
for i in features:   #nan数据都=mean
    if(df[i].isnull().sum())>0:
        pp=df[i].values.reshape(-1,1)
        imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')
        imp_mean=imp_mean.fit_transform(pp)
        df[i]=imp_mean

scale = MinMaxScaler().fit(df[features])   ## scale到range 0-1
dataScale = scale.transform(df[features])

##主成分分析
pca = PCA(n_components=10)
# 对样本进行降维
reduced_x = pca.fit_transform(dataScale)
print(reduced_x)
# # 可视化
# red_x, red_y = [], []
# blue_x, blue_y = [], []
# green_x, green_y = [], []
# for i in range(len(reduced_x)):
#     if y[i] == 0:
#         red_x.append(reduced_x[i][0])
#         red_y.append(reduced_x[i][1])
#     elif y[i] == 1:
#         blue_x.append(reduced_x[i][0])
#         blue_y.append(reduced_x[i][1])
#     else:
#         green_x.append(reduced_x[i][0])
#         green_y.append(reduced_x[i][1])
# plt.scatter(red_x, red_y, c='r', marker='x')
# plt.scatter(blue_x, blue_y, c='b', marker='D')
# plt.scatter(green_x, green_y, c='g', marker='.')
# plt.show()
#
kmeans = KMeans(n_clusters = 4,
    random_state=123).fit(reduced_x)
print('构建的K-Means模型为：\n',kmeans)
# result = kmeans.predict([[1.5,1.5,1.5,1.5]])
# print('花瓣花萼长度宽度全为1.5的鸢尾花预测类别为：', result[0])
# print('聚类结果为：', kmeans.labels_)
#
# from sklearn.metrics import calinski_harabaz_score
# for i in range(2,7):
#     ##构建并训练模型
#     kmeans = KMeans(n_clusters = i,random_state=123).fit(iris_data)
#     score = calinski_harabaz_score(iris_data,kmeans.labels_)
#     print('iris数据聚%d类calinski_harabaz指数为：%f'%(i,score))
