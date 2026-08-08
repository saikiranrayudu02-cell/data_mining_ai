from app.ml.algorithms.base import BaseClassifier
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Tuple, Union

class J48Node:
    """
    Represent a node in the J48 (C4.5) decision tree.
    Supports nominal multiway splits and continuous binary splits (e.g. column <= threshold).
    """
    def __init__(
        self,
        feature: Optional[str] = None,
        is_continuous: bool = False,
        threshold: Optional[float] = None,
        is_leaf: bool = False,
        prediction: Optional[Any] = None,
        probabilities: Optional[Dict[Any, float]] = None,
        total_instances: float = 0.0,
        error_instances: float = 0.0
    ):
        self.feature = feature
        self.is_continuous = is_continuous
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.probabilities = probabilities
        self.total_instances = total_instances
        self.error_instances = error_instances
        self.children: Dict[Any, J48Node] = {} # Nominal value -> child or boolean (True/False) -> child

class J48Classifier(BaseClassifier):
    """
    Custom J48 (C4.5) Decision Tree Classifier utilizing Gain Ratio.
    Supports continuous and nominal columns, and basic subtree pruning.
    """
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(hyperparameters)
        self.min_instances = self.hyperparameters.get("min_instances", 2)
        self.confidence_threshold = self.hyperparameters.get("confidence_threshold", 0.25)
        self.max_depth = self.hyperparameters.get("max_depth", None)
        self.root: Optional[J48Node] = None
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

    def _split_entropy_and_gain(self, X: pd.DataFrame, y: pd.Series, feature: str, is_numeric: bool) -> Tuple[float, float, Optional[float]]:
        """
        Evaluate splits and compute Information Gain, Split Information, and threshold.
        Returns: (gain, split_info, threshold)
        """
        total = len(y)
        if total == 0:
            return 0.0, 0.0, None
            
        base_entropy = self._entropy(y)
        
        if is_numeric:
            sorted_idx = X[feature].argsort()
            X_sorted = X[feature].iloc[sorted_idx]
            
            best_gain = -1.0
            best_split_info = 0.0
            best_threshold = None
            
            unique_vals = X_sorted.unique()
            if len(unique_vals) <= 1:
                return 0.0, 0.0, None
                
            thresholds = [(unique_vals[i] + unique_vals[i+1]) / 2.0 for i in range(len(unique_vals) - 1)]
            
            for t in thresholds:
                left_mask = X[feature] <= t
                right_mask = ~left_mask
                
                left_count = left_mask.sum()
                right_count = right_mask.sum()
                
                if left_count < self.min_instances or right_count < self.min_instances:
                    continue
                    
                w_left = left_count / total
                w_right = right_count / total
                
                gain = base_entropy - (w_left * self._entropy(y[left_mask]) + w_right * self._entropy(y[right_mask]))
                split_info = - (w_left * np.log2(w_left) + w_right * np.log2(w_right)) if w_left > 0 and w_right > 0 else 0.0
                
                if gain > best_gain:
                    best_gain = gain
                    best_split_info = split_info
                    best_threshold = t
                    
            return max(0.0, best_gain), best_split_info, best_threshold
        else:
            values = X[feature].unique()
            if len(values) <= 1:
                return 0.0, 0.0, None
                
            weighted_entropy = 0.0
            split_info = 0.0
            
            for val in values:
                mask = X[feature] == val
                count = mask.sum()
                weight = count / total
                weighted_entropy += weight * self._entropy(y[mask])
                if weight > 0:
                    split_info -= weight * np.log2(weight)
                
            gain = base_entropy - weighted_entropy
            return max(0.0, gain), split_info, None

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> J48Node:
        total_len = float(len(y))
        
        if len(y.unique()) <= 1:
            pred = y.iloc[0] if len(y) > 0 else "unknown"
            probs = {c: 1.0 if c == pred else 0.0 for c in self.classes_}
            return J48Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=0.0)
            
        if len(y) < self.min_instances or (self.max_depth is not None and depth >= self.max_depth):
            pred = y.mode().iloc[0] if not y.empty else "unknown"
            counts = y.value_counts(normalize=True).to_dict()
            probs = {c: float(counts.get(c, 0.0)) for c in self.classes_}
            errs = total_len - float((y == pred).sum())
            return J48Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=errs)
            
        best_feature = None
        best_ratio = -1.0
        best_threshold = None
        is_continuous_split = False
        
        for feature in X.columns:
            is_numeric = pd.api.types.is_numeric_dtype(X[feature])
            gain, split_info, t = self._split_entropy_and_gain(X, y, feature, is_numeric)
            
            ratio = gain / split_info if split_info > 0 else 0.0
                
            if ratio > best_ratio:
                best_ratio = ratio
                best_feature = feature
                best_threshold = t
                is_continuous_split = is_numeric

        if best_feature is None or best_ratio <= 0.0:
            pred = y.mode().iloc[0] if not y.empty else "unknown"
            counts = y.value_counts(normalize=True).to_dict()
            probs = {c: float(counts.get(c, 0.0)) for c in self.classes_}
            errs = total_len - float((y == pred).sum())
            return J48Node(is_leaf=True, prediction=pred, probabilities=probs, total_instances=total_len, error_instances=errs)

        node = J48Node(feature=best_feature, is_continuous=is_continuous_split, threshold=best_threshold, total_instances=total_len)
        
        if is_continuous_split:
            left_mask = X[best_feature] <= best_threshold
            right_mask = ~left_mask
            
            node.children[True] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.children[False] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        else:
            unique_vals = X[best_feature].unique()
            remaining_X = X.drop(columns=[best_feature])
            for val in unique_vals:
                mask = X[best_feature] == val
                node.children[val] = self._build_tree(remaining_X[mask], y[mask], depth + 1)
                
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "J48Classifier":
        self.classes_ = sorted(list(y.astype(str).unique()))
        
        # Calculate root candidate attribute entropy statistics
        base_entropy = self._entropy(y)
        self.entropy_stats_ = []
        for col in X.columns:
            is_num = pd.api.types.is_numeric_dtype(X[col])
            gain, split_info, t = self._split_entropy_and_gain(X, y, col, is_num)
            ratio = gain / split_info if split_info > 0 else 0.0
            self.entropy_stats_.append({
                "attribute_name": col,
                "entropy": round(base_entropy, 4),
                "info_gain": round(gain, 4),
                "split_info": round(split_info, 4),
                "gain_ratio": round(ratio, 4)
            })
            
        self.root = self._build_tree(X, y)
        self.is_trained = True
        return self

    def _predict_row(self, node: J48Node, row: pd.Series) -> Tuple[Any, Dict[Any, float]]:
        if node.is_leaf:
            return node.prediction, node.probabilities or {}
            
        val = row[node.feature]
        
        if node.is_continuous:
            branch = (float(val) <= node.threshold)
            return self._predict_row(node.children[branch], row)
        else:
            val_str = str(val)
            if val_str in node.children:
                return self._predict_row(node.children[val_str], row)
            else:
                for child in node.children.values():
                    return self._predict_row(child, row)
                return "unknown", {c: 0.0 for c in self.classes_}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.root is None:
            raise RuntimeError("J48 model has not been trained yet")
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self._predict_row(self.root, row)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.root is None:
            raise RuntimeError("J48 model has not been trained yet")
        proba_list = []
        for _, row in X.iterrows():
            _, probs = self._predict_row(self.root, row)
            proba_row = [probs.get(c, 0.0) for c in self.classes_]
            proba_list.append(proba_row)
        return np.array(proba_list)

    def get_weka_tree_text(self) -> str:
        """
        Generate ASCII indented WEKA Explorer-style tree text representation.
        """
        if not self.root:
            return "J48 pruned tree\n------------------\n(Empty tree)"
            
        lines = ["J48 pruned tree", "------------------", ""]
        
        def _recurse(node: J48Node, indent: str):
            if node.is_leaf:
                err_str = f"/{node.error_instances:g}" if node.error_instances > 0 else ""
                return f": {node.prediction} ({node.total_instances:g}{err_str})"
                
            if node.is_continuous:
                thresh_str = f"{node.threshold:.2f}".rstrip('0').rstrip('.')
                
                # True branch (<= threshold)
                left_cond = f"{node.feature} <= {thresh_str}"
                if node.children[True].is_leaf:
                    leaf_str = _recurse(node.children[True], indent)
                    lines.append(f"{indent}{left_cond}{leaf_str}")
                else:
                    lines.append(f"{indent}{left_cond}")
                    _recurse(node.children[True], indent + "|   ")
                    
                # False branch (> threshold)
                right_cond = f"{node.feature} > {thresh_str}"
                if node.children[False].is_leaf:
                    leaf_str = _recurse(node.children[False], indent)
                    lines.append(f"{indent}{right_cond}{leaf_str}")
                else:
                    lines.append(f"{indent}{right_cond}")
                    _recurse(node.children[False], indent + "|   ")
            else:
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
