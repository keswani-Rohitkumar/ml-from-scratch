import numpy as np
from collections import Counter
class LogisticRegression:

    def __init__(self, lr, num_iter, l2=0.0, class_weights = None):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.losses = []
        self.l2 = l2
        self.class_weights = class_weights

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0


        sample_weights = np.ones(n_samples)

        if self.class_weights == 'balanced':
            class_counts = Counter(y)
            n_classes = len(class_counts)
            total = len(y)
            self.class_weights = {
                cls:total / (n_classes * count)
                for cls, count in class_counts.items()
            }
        
        if isinstance(self.class_weights, dict):
            for label, weight in self.class_weights.items():
                sample_weights[y==label] = weight


        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            errors = y_predicted - y
            weighted_errors = sample_weights * errors

            dw = (1 / n_samples) * np.dot(X.T, weighted_errors)
            db = (1/n_samples) * np.sum(weighted_errors)

            if self.l2 > 0 :
                dw += self.l2 * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = - np.sum(
                sample_weights * (y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
                            )/np.sum(sample_weights)

            if self.l2 > 0:
                loss += (self.l2 / 2) * np.sum(self.weights ** 2)

            self.losses.append(loss)

    def predict_proba(self,X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_predicted_proba = self.predict_proba(X)

        return np.where(y_predicted_proba >=0.5, 1, 0)