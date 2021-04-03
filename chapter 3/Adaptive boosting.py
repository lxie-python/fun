# 每轮迭代中会在训练集上产生一个新的学习器，然后使用该学习器对所有样本进行预测，以评估每个样本的重要性(Informative)。
# 换句话来讲就是，算法会为每个样本赋予一个权重，每次用训练好的学习器标注/预测各个样本，如果某个样本点被预测的越正确，则将其权重降低；
# 否则提高样本的权重。权重越高的样本在下一个迭代训练中所占的比重就越大，也就是说越难区分的样本在训练过程中会变得越重要；
# 整个迭代过程直到错误率足够小或者达到一定的迭代次数为止。
# Adaboost算法可以简述为三个步骤：
# 首先，初始化训练数据的权值分布D1。假设有N个训练样本数据，则每一个训练样本最开始时，都被赋予相同的权值：w1=1/N。
# 然后，训练弱分类器hi。具体训练过程：如果某个训练样本点，被弱分类器hi准确地分类，那么在构造下一个训练集中，它对应的权值要减小；
#      相反，如果某个训练样本点被错误分类，那么它的权值就应该增大。权值更新过的样本集被用于训练下一个分类器。
# 最后，将各个训练得到的弱分类器组合成一个强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，
#      使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。
#      换而言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn import model_selection
from sklearn.utils import shuffle

housing_data=datasets.load_boston()
#shuffle the data
X,y=shuffle(housing_data.data,housing_data.target,random_state=7)
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=7)

regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
evs=explained_variance_score(y_test,y_pred)
print('\nADABOOST REGRESSOR')
print('Mean squared error=',round(mse,2))
print('Explained variance score=',round(evs,2))

feature_importance=regressor.feature_importances_
feature_names=housing_data.feature_names
feature_importance=100*(feature_importance/max(feature_importance))
index_sorted=np.flipud(np.argsort(feature_importance))
pos=np.arange(index_sorted.shape[0])+0.5

plt.figure()
plt.bar(pos,feature_importance[index_sorted],align='center')
plt.xticks(pos,feature_names[index_sorted])
plt.ylabel('Relative importance')
plt.title('Feature importance using Adaboost regressor')
plt.show()