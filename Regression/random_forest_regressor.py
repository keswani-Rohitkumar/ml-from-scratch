import numpy as np
from collections import Counter
from decision_tree_regressor import DecisionTreeRegressor

class RandomForest:

    def __init__(self, n_trees, max_depth, min_samples_split, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    
    def fit(self, X, y):

        self.trees = []
        n_samples, n_features_total = X.shape

        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features_total))
        
        for _ in range(self.n_trees):

            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            feat_idxs = np.random.choice(n_features_total, self.n_features, replace=False)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feat_idxs], y_sample)

            self.trees.append((tree, feat_idxs))
        print(self.trees)
    
    def predict(self, X):

        tree_preds = []
        for tree, feat_idxs in self.trees :
            preds = tree.predict(X[:, feat_idxs])
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds).T
        y_pred = np.mean(tree_preds, axis=1)

        return y_pred
    


# --- Example Usage ---
if __name__ == "__main__":
    X = np.array([[2],
              [4],
              [6],
              [8],
              [10]])
    y = np.array([5, 6, 7, 8, 9])

    # Create and train Random Forest
    rf = RandomForest(n_trees=3, max_depth=3, min_samples_split=2)
    rf.fit(X, y)

    # Make predictions
    preds = rf.predict(X)
    print("Predictions:", preds)