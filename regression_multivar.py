import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

input_file='data_multivar_regr.txt'
data=np.loadtxt(input_file,delimiter=',')
X,y=data[:,:-1],data[:,-1]

num_training=int(0.8*len(X))
X_train,y_train=X[:num_training,:],y[:num_training]
X_test,y_test=X[num_training:,:],y[num_training:]

regressor1=linear_model.LinearRegression()
regressor1.fit(X_train,y_train)

polynomial=PolynomialFeatures(degree=10)
X_train_tranformed=polynomial.fit_transform(X_train)
regressor2=linear_model.LinearRegression()
regressor2.fit(X_train_tranformed,y_train)

data_points=[[7.75,6.45,5.56]]
poly_data_points=polynomial.fit_transform(data_points)
print('Linear regression:',regressor1.predict(data_points))
print('Polynomial regression:',regressor2.predict(poly_data_points))

