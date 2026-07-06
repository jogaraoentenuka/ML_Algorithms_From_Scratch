import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class kNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
        
        
    def _predict(self, x): 
        # compute distances  
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get the knns
        k_indices = np.argsort(distances)[:self.k] # argsort() returns the indices that would sort an array, rather than the sorted array itself.
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class label
        # most_common = np.bincount(k_nearest_labels).argmax() -> fast but works with only integer classes
        most_common = Counter(k_nearest_labels).most_common()[0][0] # works with any type of class labels
        return most_common