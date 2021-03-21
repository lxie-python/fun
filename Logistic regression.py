import numpy as np
from utilities import visualize_classifier
from sklearn import linear_model

X=np.array([[3.1,7.2],[4,6.7],[2.9,8],
            [5.1,4.5],[6,5],[5.6,5],
            [3.3,0.5],[3.9,0.9],[2.8,1],
            [0.5,3.4],[1,4],[0.6,4.9]])
y=np.array([0,0,0,1,1,1,2,2,2,3,3,3])

classifier=linear_model.LogisticRegression(solver='liblinear',C=1)
#C imposes a certain penalty on misclassification, so the algorithm
#customizes more to the training data. You should be careful with this parameter, because if
#you increase it by a lot, it will overfit to the training data and it won't generalize well.
classifier.fit(X,y)
visualize_classifier(classifier,X,y)