# one of the best things about random forests is that they do not overfit.
# the nodes are split successively and the best thresholds are chosen to reduce the entropy at each level.
# This split considers the random subset of the features. This may increse the bias, but the variance decreases
# because of averaging
# Extremely Random Forests take randomness to the next level, the thresholds are chosen random too.
# These randomly generated thresholds reduce the variance of the model even further.
# Hence the boundaries obtained using Extremely Random Forests tend to be smoother than Random Forests.
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier
# Argument parser
# def build_arg_parser():
#     parser=argparse.ArgumentParser(description='Classify data using Ensenble Learning techniques')
#     parser.add_argument('--classifier-type',dest='classifier_type',required=True,choices=['rf','erf'],
#                         help='Typer of classifier to use; can be either rf or erf')
#     return parser
#
# if __name__=='__main__':
#     args=build_arg_parser().parse_args()
#     classifier_type=args.classifier_type

input_file='data_random_forests.txt'
data=np.loadtxt(input_file,delimiter=',')
X,y=data[:,:-1],data[:,-1]
class_0=np.array(X[y==0])
class_1=np.array(X[y==1])
class_2=np.array(X[y==2])
    #visualize input data
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],s=75,facecolor='white',edgecolors='black',linewidths=1,marker='s')
plt.scatter(class_1[:,0],class_1[:,1],s=75,facecolor='white',edgecolors='black',linewidths=1,marker='o')
plt.scatter(class_2[:,0],class_2[:,1],s=75,facecolor='white',edgecolors='black',linewidths=1,marker='^')
plt.title('input data')

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.25,random_state=5)

params={'n_estimators':100,'max_depth':4,'random_state':1}

# classifier=RandomForestClassifier(**params)

classifier=ExtraTreesClassifier(**params)

classifier.fit(X_train,y_train)
visualize_classifier(classifier,X_train,y_train)
y_pred=classifier.predict(X_test)
visualize_classifier(classifier,X_test,y_test)

class_name=['class 0','class 1','class 2']
print('\n'+'#'*40)
print('\nClassifier performance on training dataset\n')
print(classification_report(y_train,classifier.predict(X_train),target_names=class_name))
print('\n'+'#'*40)
print('\nClassification performance on testing dataset')
print(classification_report(y_test,classifier.predict(X_test),target_names=class_name))
plt.show()

#compute confidence
test_datapoints=np.array([[5,5],[3,6],[6,4],[7,2],[4,4],[5,2]])
print('\nConfidence measure:')
for datapoint in test_datapoints:
    probabilities=classifier.predict_proba([datapoint])[0]
    predicted_class='Class-'+str(np.argmax(probabilities))
    print('\nDatapoint:',datapoint)
    print('Predicted class:',predicted_class)