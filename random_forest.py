import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple
from tqdm import tqdm
#ouo

class DecisionTree:
    def __init__(self, max_depth=5, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.features = X.columns.tolist()
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_tree(row, self.tree) for _, row in X.iterrows()])

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        if len(set(y)) == 1 or depth == self.max_depth or X.empty:
            return {'type': 'leaf', 'value': np.mean(y)}

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return {'type': 'leaf', 'value': np.mean(y)}

        left_mask = X.iloc[:, feature_idx] <= threshold
        X_left, y_left = X[left_mask].reset_index(drop=True), y[left_mask]
        X_right, y_right = X[~left_mask].reset_index(drop=True), y[~left_mask]

        return {
            'type': 'node',
            'feature_index': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }

    def _predict_tree(self, row, node):
        if node['type'] == 'leaf':
            return node['value']
        if row.iloc[node['feature_index']] <= node['threshold']:
            return self._predict_tree(row, node['left'])
        else:
            return self._predict_tree(row, node['right'])

    def _best_split(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[int, float]:
        best_mse = float('inf')
        best_feat = None
        best_thresh = None
        num_features=X.shape[1]
        if self.max_features=='sqrt':
            num_features= int(np.sqrt(X.shape[1]))
        elif self.max_features=='log2':
            num_features=int(np.log2(X.shape[1]))
        elif isinstance(self.max_features,int):
            num_features= self.max_features
        else: 
            num_features=X.shape[1]


        feature_indices = np.random.choice(
            X.shape[1],
            num_features,
            replace=False
        )

        for feature_idx in feature_indices:
            values = X.iloc[:, feature_idx].sort_values().unique()
            for i in range(1, len(values)):
                thresh = (values[i - 1] + values[i]) / 2
                left = y[X.iloc[:, feature_idx] <= thresh]
                right = y[X.iloc[:, feature_idx] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (len(left) * np.var(left) + len(right) * np.var(right)) / len(y)
                if mse < best_mse:
                    best_mse, best_feat, best_thresh = mse, feature_idx, thresh

        return best_feat, best_thresh


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        self.n_trees = n_trees # how many trees in total
        self.max_depth = max_depth
        self.max_features = max_features # how many feature is used in trees
        self.trees = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Training trees..."):
            # bootstrap sampling:
            # randomly select index of the original data (may appear repeated one)
            idxs = np.random.choice(len(X), len(X), replace=True)

            # select the bootstrap sample
            X_sample = X.iloc[idxs].reset_index(drop=True)
            y_sample = y[idxs]

            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions= np.array([tree.predict(X) for tree in self.trees])
        # take the mean of each trees 
        return predictions.mean(axis=0)
