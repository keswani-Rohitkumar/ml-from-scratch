import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:

    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X,y)
    

    def build_tree(self, X, y, depth = 0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        
        best_feat, best_thresh = self._best_split(X,y)

        if best_feat is None:
            return Node(value=self._most_common_label(y))
        
        left_idx, right_idx = self._split(X[:, best_feat], best_thresh)
        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feat, best_thresh, left, right)
    
    def _best_split(self, X,y):
        best_gain = -1
        split_idx, split_thresh = None, None

        num_features = X.shape[1]

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx, right_idx = self._split(X[:, feature], threshold)
                if len(left_idx) ==0 or len(right_idx) == 0:
                    continue
                    
                gain = self._information_gain(y, y[left_idx], y[right_idx])

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def _split(self, feature_column, threshold):
        left = np.argwhere(feature_column<=threshold).flatten()
        right = np.argwhere(feature_column > threshold).flatten()
    
        return left, right

    def _information_gain(self, parent,left,right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        return self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))
    
    def _gini(self, y):
        """Compute Gini impurity for labels y"""
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        else:
            return self._traverse_tree(x, node.right)
        
if __name__ == "__main__":
    # Simple dataset
    X = np.array([
        [2, 3],
        [1, 5],
        [4, 2],
        [6, 1],
        [7, 3]
    ])
    y = np.array([0, 0, 1, 1, 1])

    # Create and train tree
    clf = DecisionTree(max_depth=3, min_samples_split=2)
    clf.fit(X, y)

    # Predict
    preds = clf.predict(X)
    print("Predictions:", preds)