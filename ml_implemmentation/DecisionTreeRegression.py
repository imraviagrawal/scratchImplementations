import numpy as np

class DecisionTreeRegression:
    def __init__(self, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = {}
        self.tree_node = {}

    def train(self, X, y):
        self.root = self.find_best_split(X, y)
        self.tree_node = self.build_tree()

    def build_tree(self, node, depth):
        left, y_left, right, y_right = node["left"], node["y_left"], node["right"], node["y_right"]
        # use get_terminal_node for terminal code
        if len(y_left) == 0 or len(y_right) == 0:
            node["left_child"] = node["right_child"] = self.get_terminal_node(y_left + y_right)
            return node

        # check max_depth
        if self.max_depth <= depth:
            node["left_child"] = self.get_terminal_node(y_left)
            node["right_child"] = self.get_terminal_node(y_right)
            return node

        if len(left) < self.min_samples:
            node["left_child"] = self.get_terminal_node(y_left)
        else:
            left_node = self.find_best_split(left, y_left)
            node["left_child"] = self.build_tree(left_node, depth+1)

        if len(right)<self.min_samples:
            node["right_child"] = self.get_terminal_node(y_right)
        else:
            right_node = self.find_best_split(right, y_right)
            node["right_child"] = self.build_tree(right_node, depth+1)

        return node


    def get_terminal_node(self, y):
        return self.get_y_hat(y)

    def find_best_split(self, X, y):
        best_split, best_feature_idx, best_cost, best_threshold = None, np.inf, np.inf, np.inf
        n_samples, m_features = X.shape
        for features_idx in range(m_features):
            for n_idx in range(n_samples):
                threshold = X[n_idx, features_idx]

                split = self.split_dataset(X, y, features_idx, threshold)
                # calcuate the cost
                curr_cost = self.get_cost(split["y_left"], split["y_right"])
                if curr_cost < best_cost:
                    best_threshold = threshold
                    best_cost = curr_cost
                    best_feature_idx = features_idx
                    best_split = split

        result = {"best_feature_idx": best_feature_idx,
                  "best_cost": best_cost,
                  "best_threshold": best_threshold,
                  'left': split["left"],
                  'y_left': split["y_left"],
                  'right': split["right"],
                  'y_right': split["y_right"],}
        return result

    def split_dataset(self, X, y, feature_idx, threshold):
        """
        Splits dataset X into two subsets, according to a given feature
        and feature threshold.

        Args:
            X: 2D numpy array with data samples
            y: 1D numpy array with labels
            feature_idx: int, index of feature used for splitting the data
            threshold: float, threshold used for splitting the data

        Returns:
            splits: dict containing the left and right subsets
            and their labels
        """

        left_idx = np.where(X[:, feature_idx] < threshold)
        right_idx = np.where(X[:, feature_idx] >= threshold)

        left_subset = X[left_idx]
        y_left = y[left_idx]

        right_subset = X[right_idx]
        y_right = y[right_idx]

        splits = {
        'left': left_subset,
        'y_left': y_left,
        'right': right_subset,
        'y_right': y_right,
        }

        return splits

    def get_y_hat(self, y):
        return np.mean(y)

    def _cost(self, y, y_hat):
        n_samples = y.shape[0]
        distance = np.sum((y - y_hat)**2)
        return (1/n_samples)*distance

    def get_cost(self, left_y, right_y):
        left_y_hat = self.get_y_hat(left_y)
        right_y_hat = self.get_y_hat(right_y)

        left_cost = self._cost(left_y, left_y_hat)
        right_cost = self._cost(right_y, right_y_hat)

        n_left = left_cost.shape[0]
        n_right = right_cost.shape[0]
        n_total = n_left + n_right

        cost = (n_left/n_total)*(left_cost) + (n_right/n_total)*(right_cost)
        return cost

