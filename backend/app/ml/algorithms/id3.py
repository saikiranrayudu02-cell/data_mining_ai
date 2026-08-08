from app.ml.algorithms.base import BaseClassifier
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Union, Tuple

class ID3Node:
    """
    Represent a node in the ID3 decision tree.
    """
    def __init__(
        self,
        feature: Optional[str] = None,
        is_leaf: bool = False,
        prediction: Optional[Any] = None,
        probabilities: Optional[Dict[Any, float]] = None,
        total_instances: float = 0.0,
        error_instances: float = 0.0
    ):
        self.feature = feature              # Feature name we split on
        self.is_leaf = is_leaf              # Leaf flag
        self.prediction = prediction        # Leaf predicted value
        self.probabilities = probabilities  # Class probability distribution at leaf
        self.total_instances = total_instances
        self.error_instances = error_instances
        self.children: Dict[Any, ID3Node] = {} # Map nominal value -> ID3Node child

class ID3Classifier(BaseClassifier):
    """
    Custom ID3 Decision Tree Classifier utilizing Information Gain.
    """
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(hyperparameters)
        self.max_depth: Optional[int] = self.hyperparameters.get("max_depth", None)
        self.root: Optional[ID3Node] = None
        self.classes_: List[Any] = []
        self.entropy_stats_: List[Dict[str, Any]] = []

    def _entropy(self, y: pd.Series) -> float:
        counts = y.value_counts()
        total = len(y)
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts:
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _information_gain(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        base_entropy = self._entropy(y)
        values = X[feature].unique()
        total_instances = len(y)
        
        weighted_entropy = 0.0
        for val in values:
            subset_indices = X[feature] == val
            y_subset = y[subset_indices]
            weight = len(y_subset) / total_instances
            weighted_entropy += weight * self._entropy(y_subset)
            
        return base_entropy - weighted_entropy

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> ID3Node:
        total_len = float(len(y))
        
        if len(y.unique()) <= 1:
            pred = y.iloc[0] if len(y) > 0 else "unknown"
            probs = {c: 1.0 if c == pred else 0.0 for c in self.classes_}
            return ID3Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=0.0)

        if X.empty or (self.max_depth is not None and depth >= self.max_depth):
            pred = y.mode().iloc[0] if not y.empty else "unknown"
            counts = y.value_counts(normalize=True).to_dict()
            probs = {c: float(counts.get(c, 0.0)) for c in self.classes_}
            errs = total_len - float((y == pred).sum())
            return ID3Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=errs)

        best_feature = None
        best_gain = -1.0
        
        for feature in X.columns:
            gain = self._information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_feature is None or best_gain <= 0.0:
            pred = y.mode().iloc[0] if not y.empty else "unknown"
            counts = y.value_counts(normalize=True).to_dict()
            probs = {c: float(counts.get(c, 0.0)) for c in self.classes_}
            errs = total_len - float((y == pred).sum())
            return ID3Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=errs)

        node = ID3Node(feature=best_feature, total_instances=total_len)
        
        unique_vals = X[best_feature].unique()
        remaining_X = X.drop(columns=[best_feature])
        
        for val in unique_vals:
            subset_indices = X[best_feature] == val
            X_subset = remaining_X[subset_indices]
            y_subset = y[subset_indices]
            
            if len(y_subset) == 0:
                pred = y.mode().iloc[0] if not y.empty else "unknown"
                probs = {c: 1.0 if c == pred else 0.0 for c in self.classes_}
                node.children[val] = ID3Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=0.0, error_instances=0.0)
            else:
                node.children[val] = self._build_tree(X_subset, y_subset, depth + 1)
                
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ID3Classifier":
        X_nominal = X.copy().astype(str)
        y_nominal = y.copy().astype(str)
        
        self.classes_ = sorted(list(y_nominal.unique()))
        
        # Calculate root candidate entropy stats
        base_entropy = self._entropy(y_nominal)
        self.entropy_stats_ = []
        for col in X_nominal.columns:
            gain = self._information_gain(X_nominal, y_nominal, col)
            self.entropy_stats_.append({
                "attribute_name": col,
                "entropy": round(base_entropy, 4),
                "info_gain": round(gain, 4),
                "split_info": None,
                "gain_ratio": None
            })
            
        self.root = self._build_tree(X_nominal, y_nominal)
        self.is_trained = True
        return self

    def _predict_row(self, node: ID3Node, row: pd.Series) -> Tuple[Any, Dict[Any, float]]:
        if node.is_leaf:
            return node.prediction, node.probabilities or {}
            
        feature_val = str(row[node.feature])
        if feature_val in node.children:
            return self._predict_row(node.children[feature_val], row)
        else:
            for child in node.children.values():
                return self._predict_row(child, row)
            return "unknown", {c: 0.0 for c in self.classes_}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.root is None:
            raise RuntimeError("ID3 model has not been trained yet")
            
        X_nominal = X.astype(str)
        predictions = []
        for _, row in X_nominal.iterrows():
            pred, _ = self._predict_row(self.root, row)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.root is None:
            raise RuntimeError("ID3 model has not been trained yet")
            
        X_nominal = X.astype(str)
        proba_list = []
        for _, row in X_nominal.iterrows():
            _, probs = self._predict_row(self.root, row)
            proba_row = [probs.get(c, 0.0) for c in self.classes_]
            proba_list.append(proba_row)
        return np.array(proba_list)

    def get_weka_tree_text(self) -> str:
        """
        Generate ASCII indented WEKA Explorer-style tree text representation for ID3.
        """
        if not self.root:
            return "ID3 decision tree\n------------------\n(Empty tree)"
            
        lines = ["ID3 decision tree", "------------------", ""]
        
        def _recurse(node: ID3Node, indent: str):
            if node.is_leaf:
                err_str = f"/{node.error_instances:g}" if node.error_instances > 0 else ""
                return f": {node.prediction} ({node.total_instances:g}{err_str})"
                
            for val, child in node.children.items():
                cond = f"{node.feature} = {val}"
                if child.is_leaf:
                    leaf_str = _recurse(child, indent)
                    lines.append(f"{indent}{cond}{leaf_str}")
                else:
                    lines.append(f"{indent}{cond}")
                    _recurse(child, indent + "|   ")
                    
        _recurse(self.root, "")
        return "\n".join(lines)
