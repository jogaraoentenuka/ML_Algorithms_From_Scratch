import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_reg import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

def r2_val(y_test, predictions):
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ss_res / ss_tot)

mse = mse(y_test, predictions)
r2 = r2_val(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

y_pred_line = reg.predict(X_test)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_test, y_test, color=cmap(0.9), label='Test Data')
m2 = plt.scatter(X_train, y_train, color=cmap(0.5), label='Train Data')
plt.plot(X_test, y_pred_line, color='red', linewidth=2, label='Prediction')
plt.show()
                 