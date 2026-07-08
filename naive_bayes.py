import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.prior_probs = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # calculate mean, variance, and prior probabilities for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior_probs = np.zeros(n_classes, dtype=np.float64)
        
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.mean[idx, :] = X_cls.mean(axis=0)
            self.var[idx, :] = X_cls.var(axis=0)
            self.prior_probs[idx] = X_cls.shape[0] / float(n_samples)
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):  
        posteriors = []
        
        # calculate the prosterior probability for each class
        for idx, cls in enumerate(self.classes):
            prior = np.log(self.prior_probs[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
        

        
