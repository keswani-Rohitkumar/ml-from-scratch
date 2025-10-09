import numpy as np
from collections import Counter

class Node:
    
    def __init__(self, feature = None, threshold = None, left = None, right = None, * , value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeRegressor:

    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X,y):
        self.root = self._build_tree(X,y)

    def _build_tree(self, X, y , depth=0):

        num_samples, num_features = X.shape

        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return Node(value=np.mean(y))
        
        best_feat, best_thresh = self._best_split(X,y)

        if best_feat is None:
            return Node(value=np.mean(y))
        
        left_idx, right_idx = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idx, :], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth+1)

        return Node(best_feat, best_thresh, left, right)
    
    def _best_split(self, X, y):

        best_var_red = -1
        split_idx , splid_thresh = None, None
        num_features = X.shape[1]

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx, right_idx = self._split(X[:, feature], threshold)

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                var_red = self._variance_reduction(y, y[left_idx], y[right_idx])

                if var_red > best_var_red:
                    best_var_red = var_red
                    split_idx = feature
                    splid_thresh = threshold
        return split_idx, splid_thresh
    

    def _split(self, feature_column, threshold):
        left = np.argwhere(feature_column <= threshold).flatten()
        right = np.argwhere(feature_column > threshold).flatten()
        return left, right
    
    def _variance_reduction(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        return np.var(parent) - (weight_left * np.var(left) + weight_right * np.var(right))
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
# Toy regression dataset
X = np.array([[2],[4],[6],[8],[10]])
y = np.array([5, 6, 7, 8, 9])

tree = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
tree.fit(X, y)

preds = tree.predict(X)
print("Predictions:", preds)
