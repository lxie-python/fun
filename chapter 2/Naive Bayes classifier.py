#The assumption here is that the value of any given feature is independent of the value of any
#other feature. This is called the independence assumption, which is the naïve part of a Naïve
#Bayes classifier.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from utilities import visualize_classifier
import pandas as pd
from sklearn.impute import SimpleImputer

input_file= 'data_test.txt'
data=np.loadtxt(input_file,delimiter=",")
X,y=data[:,:-1],data[:,-1]

# df= pd.read_excel('data_mine.xls')
# features=df.columns[:-1]
# target=df.columns[-1]
#
# for i in features:   #nan数据都=mean
#     if(df[i].isnull().sum())>0:
#         pp=df[i].values.reshape(-1,1)
#         imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')
#         imp_mean=imp_mean.fit_transform(pp)
#         df[i]=imp_mean

# X,y=df[features].values,df[target].values

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=3)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

accuracy=100*(y_test==y_pred).sum()/X_test.shape[0]
print('Accuracy of Naive Bayes classifier=',round(accuracy,2),'%')

visualize_classifier(classifier,X,y)



