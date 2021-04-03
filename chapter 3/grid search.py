import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report

input_file='data_random_forests.txt'
data=np.loadtxt(input_file,delimiter=',')
X,y=data[:,:-1],data[:,-1]

class_0=np.array(X[y==0])
class_1=np.array(X[y==1])
class_2=np.array(X[y==2])

plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],s=75,facecolor='black',edgecolors='black',linewidths=1,marker='x')
plt.scatter(class_1[:,0],class_1[:,1],s=75,facecolor='white',edgecolors='black',linewidths=1,marker='o')
plt.title('input data')
plt.show()

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.25,random_state=5)

parameter_grid=[{'n_estimators':[100],'max_depth':[2,4,7,12,16]},{'max_depth':[4],'n_estimators':[25,50,100,
                                                                                                250]}]
metrics=['precision_weighted','recall_weighted']

for metric in metrics:
    print('\n##### Searching optimal parameters for',metric)
    classifier=model_selection.GridSearchCV(
        ExtraTreesClassifier(random_state=0),parameter_grid,cv=5,scoring=metric)
    classifier.fit(X_train,y_train)
    print('\nGrid scores for the parameter grid:')
    means=classifier.cv_results_['mean_test_score']
    params=classifier.cv_results_['params']
    for mean,param in zip(means,params):
        print(param,'-->',mean)
    print('\nBest parameters:',classifier.best_params_)

    y_pred=classifier.predict(X_test)
    print(classification_report(y_test,y_pred))