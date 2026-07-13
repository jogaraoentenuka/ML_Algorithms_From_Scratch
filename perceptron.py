import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def unit_step_function(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.activation_function = unit_step_function
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init Parameters
        self.weights = np.random.rand(n_features)
        self.bias = 0
        
        y = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iter):
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                # update weights and bias
                update = self.learning_rate * (y[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update
        
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    
# testing 

if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    p = Perceptron(learning_rate=0.01, n_iter=1000)
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    
    print("Perceptron classification accuracy:", p.accuracy(y_test, y_pred))
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k', s=100)
    plt.title("Perceptron Classification Results") 
    
    x0_1 = np.amin(X_test[:, 0])
    x0_2 = np.amax(X_test[:, 0])
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--', lw=2) 
    
    y_min = np.amin(X_test[:, 1]) - 1
    y_max = np.amax(X_test[:, 1]) + 1
    ax.set_ylim([y_min, y_max])
    
    plt.show()
    
    
    
    
    