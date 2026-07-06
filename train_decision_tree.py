from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from decision_tree import DecisionTree
from sklearn.metrics import roc_auc_score

data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def precision(y_test, y_pred):
    true_positive = np.sum((y_test == 1) & (y_pred == 1))
    false_positive = np.sum((y_test == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0  

def recall(y_test, y_pred):
    true_positive = np.sum((y_test == 1) & (y_pred == 1))
    false_negative = np.sum((y_test == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0  

def f1_score(y_test, y_pred):
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def roc_auc(y_test, y_pred):
    return roc_auc_score(y_test, y_pred)
    

def confusion_matrix(y_test, y_pred):
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])


acc = accuracy(y_test, predictions)
prec = precision(y_test, predictions)
rec = recall(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc = roc_auc(y_test, predictions)
cm = confusion_matrix(y_test, predictions) 

print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall: {rec * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC AUC Score: {roc:.4f}")
print("Confusion Matrix:", cm, sep="\n")




