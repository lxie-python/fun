# A Mixture Model is a type of probability density model where we assume that
# the data is governed by a number of component distributions.
# If these distributions are Gaussian, then the model becomes a Gaussian Mixture Model.
# 比如我想model所有people in 南美洲的shopping habits，一种做法就是我用一个model然后把数据fit进去
# 但是我们知道people in different countries shop differently，我们就可以每个弄一个model 然后把它们mix起来
# mixture model是semi-parametric，那如果你确定了每个model的function，就变成了parametric，GMM就是parametric
# The parameters of the GMM are estimated from training data using algorithms
# like Expectation–Maximization (EM) or Maximum A-Posteriori (MAP) estimation.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

iris=datasets.load_iris()
f=StratifiedKFold(n_splits=5,shuffle=True)
indices=f.split(iris.data,iris.target) #indices现在是一个迭代器

train_index,test_index=next(iter(indices))
X_train,y_train=iris.data[train_index],iris.target[train_index]
X_test,y_test=iris.data[test_index],iris.target[test_index]
num_classes=len(np.unique(y_train))#一共有几类
# The init_params parameter controls the parameters that need to be updated during the training process.
# wc means weights and covariance parameters will be updated during training.
classifier=GaussianMixture(n_components=num_classes,covariance_type='full')
# Initialize the means of the classifier:
classifier.means_=np.array([X_train[y_train==i].mean(axis=0)
                            for i in range(num_classes)])

classifier.fit(X_train)

