import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_reg import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

breast_cancer_data = datasets.load_breast_cancer()

X = breast_cancer_data.data
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)


y = breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classifier = LogisticRegression(learning_rate=0.01, num_iterations=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

accuracy_score = accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy_score * 100:.2f}%')

# ---------------------------------------------
# Visualize Decision Boundary
# ---------------------------------------------

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array(classifier.predict(grid))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")

plt.scatter(
    X_test[:,0],
    X_test[:,1],
    c=y_test,
    cmap="RdBu",
    edgecolors="k"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Logistic Regression Decision Boundary")
plt.show()