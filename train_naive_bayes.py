import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Naive Bayes classification accuracy:", accuracy(y_test, y_pred))