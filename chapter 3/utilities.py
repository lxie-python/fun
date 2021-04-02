import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier,X,y):
    min_x,max_x=X[:,0].min()-1,X[:,0].max()+1
    min_y,max_y=X[:,1].min()-1,X[:,1].max()+1
    mesh_step_size=0.01
    x_vals,y_vals=np.meshgrid(np.arange(min_x,max_x,mesh_step_size),np.arange(min_y,max_y,mesh_step_size))

    output=classifier.predict(np.c_[x_vals.ravel(),y_vals.ravel()])
    output=output.reshape(x_vals.shape)

    plt.figure()
    plt.pcolormesh(x_vals,y_vals,output,shading='auto',cmap=plt.cm.Greys)

    plt.scatter(X[:,0],X[:,1],c=y,s=75,edgecolors='black',linewidths=1,cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(),x_vals.max())
    plt.ylim(y_vals.min(),y_vals.max())

    plt.xticks((np.arange(int(X[:,0].min()-1),int(X[:,0].max()+1),1)))
    plt.yticks((np.arange(int(X[:,1].min()-1),int(X[:,1].max()+1),1)))

    plt.show()


