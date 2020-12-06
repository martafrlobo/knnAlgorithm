import numpy as np
from collections import Counter
import scipy as sp


"""euclidean distance"""
def distance(X_train,X_test):
    point1=np.array(X_train)
    point2=np.array(X_test)
    return np.linalg.norm(point1-point2)

"""defining accuracy"""
def accuracy(y_test, predictions):
    accuracy = np.sum(y_test == predictions) / len(y_test)
    return accuracy  

class KNN():
    
    def __init__(self,k=3):
        self.k=k
    
    def train(self,X,y):
        n_samples = X.shape[0]
        # number of neighbors can't be larger then number of samples
        if self.k > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        self.X_train=X
        self.y_train=y

    """predict new samples, this can have multiple samples"""
    def test(self,X):
        labels=[self.predict(x) for x in X]
        return np.array(labels)
    

    def predict(self,x):
        """compute distances"""
        distances=[distance(x,x_train) for x_train in self.X_train]
        """get neighbors"""
        index=np.argsort(distances)[:self.k]
        nearest_neighbors=[self.y_train[i] for i in index]
        """most common values"""
        most_common=Counter(nearest_neighbors).most_common(1)
        return most_common[0][0]