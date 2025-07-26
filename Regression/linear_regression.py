import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, num_iter=1000, l1_lambda = 0.1, l2_lambda = 0.1):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.losses = []
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.num_iter):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            if self.l1_lambda > 0:
                dw+= self.l1_lambda *np.sign(self.weights)
            
            if self.l2_lambda > 0:
                dw+= self.l2_lambda * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = np.mean(error ** 2)
            if self.l1_lambda > 0:
                loss += self.l1_lambda * np.sum(np.abs(self.weights))
            if self.l2_lambda > 0:
                loss+= self.l2_lambda * np.sum(np.abs(self.weights) ** 2)

            print(self.weights, self.bias)
            self.losses.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
        


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias