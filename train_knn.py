import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import kNN

cmap_pred = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
cmap_train = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = kNN(k=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:,2], X_test[:,3], c=y_pred, cmap=cmap_pred, edgecolor='k', s=50)
plt.scatter(X_train[:,2], X_train[:,3], c=y_train, cmap=cmap_train, edgecolor='k', s=50, alpha=0.5)
plt.show()
