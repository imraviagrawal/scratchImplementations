import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=4, min_samples=5):
        self.rootdict = {}
        self.treedict = {}
        self.max_depth = max_depth
        self.min_samples = min_samples

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature_idx, best_threshold, best_cost, best_splits = np.inf, np.inf, np.inf, None

        for i in range(n_features):
            for d in range(n_samples):
                threshold = X[d, i] # new threshould
                splits = self.split_dataset(X, y, i, threshold)

                # compute cost
                curr_cost = self.compute_cost(splits)

                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_feature_idx = i
                    best_threshold = threshold
                    best_splits = splits

        best_split_params = {
            'feature_idx': best_feature_idx,
            'threshold': best_threshold,
            'cost': best_cost,
            'left_X': best_splits['left_X'],
            'left_y': best_splits['left_y'],
            'right_X': best_splits['right'],
            'right_y': best_splits['right_y'],
        }
        return best_split_params

    def split_dataset(self, X, y, feature_idx, threshold):
        left_index = np.where(X[:, feature_idx]<threshold)
        right_index = np.where(X[:, feature_idx]>=threshold)

        left_X = X[left_index]
        right_X = X[right_index]

        left_y = y[left_index]
        right_y = y[right_index]

        splits = {"left": left_X, "y_left":left_y,\
                  "right":right_X, "y_right":right_y}
        return splits

    def compute_cost(self, splits):
        left_y = splits["left_y"]
        right_y = splits["left_y"]

        n_left = left_y.shape[0]
        n_right = right_y.shape[0]

        total = n_left+n_right
        gini_left, gini_right = self.gini_impurity(left_y, right_y, n_left, n_right)
        total_loss = (n_left/total)*gini_left + (n_right/total)*gini_right
        return total_loss

    def gini_impurity(self, left_y, right_y, n_left, n_right):
        score_left, score_right = 0, 0
        if n_left != 0:
            for c in range(self.n_classes):
                left_frac = np.where(left_y==c)[0].shape[0]/len(n_left)
                score_left += left_frac*left_frac
        gini_left = 1-score_left

        if n_right != 0:
            for c in range(self.n_classes):
                right_frac = np.where(right_y == c)[0].shape[0] / len(n_right)
                score_right += right_frac * right_frac
        gini_right = 1-score_right
        return gini_left, gini_right

    def build_tree(self, node_dict, depth):
        left_samples = node_dict['left']
        right_samples = node_dict['right']
        y_left_samples = node_dict['y_left']
        y_right_samples = node_dict['y_right']

        if len(left_samples) == 0 or len(right_samples) == 0:
            node_dict["left_child"] = node_dict["right_child"] = self.create_terminal_node(y_left_samples+y_right_samples)
            return None

        # max_depth
        if depth >= self.max_depth:
            node_dict["left_child"] = self.create_terminal_node(y_left_samples)
            node_dict["right_child"] = self.create_terminal_node(y_right_samples)
            return None

        if len(y_right_samples)<self.min_samples:
            node_dict["right_child"] = self.create_terminal_node(y_right_samples)
        else:
            node_dict["right_child"] = self.find_best_split(right_samples, y_right_samples)
            self.build_tree(node_dict["right_child"], depth+1)

        if len(y_left_samples)<self.min_samples:
            node_dict["left_child"]=self.create_terminal_node(y_left_samples)
        else:
            node_dict["left_child"]= self.find_best_split(left_samples, y_left_samples)
            self.build_tree(node_dict["left_child"], depth+1)

        return node_dict

    def create_terminal_node(self, y):
        # send the max y count as label
        counts = np.unique(y, return_counts=True)
        index = np.argmax(counts[1])
        return counts[0][index]

    def train(self, X, y):
        self.n_classes = len(set(y))

        # find first split
        self.rootdict = self.find_best_split(X, y)

        # build tree:
        self.treedict = self.build_tree(self.rootdict, 1)

    def predict(self, X, node):
        """
        Predicts the class for a given input example X.

        Args:
            X: 1D numpy array, input example
            node: dict, representing trained decision tree

        Returns:
            prediction: int, predicted class
        """
        feature_idx = node['feature_idx']
        threshold = node['threshold']

        if X[feature_idx] < threshold:
            if isinstance(node['left_child'], (int, np.integer)):
                return node['left_child']
            else:
                prediction = self.predict(X, node['left_child'])
        elif X[feature_idx] >= threshold:
            if isinstance(node['right_child'], (int, np.integer)):
                return node['right_child']
            else:
                prediction = self.predict(X, node['right_child'])

        return prediction

if __name__=="__main__":
    dt = DecisionTree(4, 5)
    np.random.seed(42)
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 3, (50, 1))
    dt.train(X, y)