import numpy as np
from collections import Counter

class KNN: 

    def __init__(self,k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def _euclidean_distance(self, x1,x2):
        return np.sqrt(np.sum(x1-x2)**2)
    
    def predict(self,X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self,x):

        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        print(Counter(k_nearest_labels))
        print(most_common)
        return most_common[0][0]


# Example dataset (2D points)
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])
y_train = np.array([0, 0, 0, 1, 1, 1])  # labels: 0 or 1

# Test data
X_test = np.array([
    [4, 5],
    [7, 7]
])

# Create and test the model
model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:", predictions)

