# Entropy is basically a measure of uncertainty. As we move from the root node towards the leaf nodes
# we need to reduce the uncertainty
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier

input_data='data_decision_trees.txt'
data=np.loadtxt(input_data,delimiter=',')
X,y=data[:,:-1],data[:,-1]
class_0=np.array(X[y==0])
class_1=np.array(X[y==1])
#visualize the input data
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],s=75,facecolors='black',edgecolors='black',linewidth=1,marker='x')
plt.scatter(class_1[:,0],class_1[:,1],s=75,facecolor='blue',edgecolors='blue',linewidths=1,marker='o')
plt.title('Input data')

#decision tree
params={'random_state':1,'max_depth':4}
classifier=DecisionTreeClassifier(**params)
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=5)
classifier.fit(X_train,y_train)
visualize_classifier(classifier,X_train,y_train)

y_test_pred=classifier.predict(X_test)
visualize_classifier(classifier,X_test,y_test)
#performance
class_names=['class 0','class 1']
print('\n'+'#'*40)
print('\nclassifier performance on training dataset\n')
print(classification_report(y_train,classifier.predict(X_train),target_names=class_names))
print('\n'+'#'*40)
print('\nclassifier performance on testing dataset\n')
print(classification_report(y_test,classifier.predict(X_test),target_names=class_names))
plt.show()

