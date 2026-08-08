from app.ml.algorithms.base import BaseClassifier
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

class NaiveBayesClassifier(BaseClassifier):
    """
    Wrapper around scikit-learn's GaussianNB.
    Includes built-in LabelEncoding pipeline for nominal attributes.
    """
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        super().__init__(hyperparameters)
        var_smoothing = self.hyperparameters.get("var_smoothing", 1e-9)
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.nominal_encoders: Dict[str, LabelEncoder] = {}
        self.classes_: List[Any] = []

    def _preprocess(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Encode nominal columns to integers using LabelEncoder.
        """
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == object or pd.api.types.is_string_dtype(X_encoded[col]):
                if is_training:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].fillna("missing").astype(str))
                    self.nominal_encoders[col] = le
                else:
                    le = self.nominal_encoders.get(col)
                    if le is not None:
                        # Handle unseen classes during prediction by mapping to first class
                        vals = X_encoded[col].fillna("missing").astype(str)
                        known_classes = set(le.classes_)
                        vals_mapped = vals.apply(lambda x: x if x in known_classes else le.classes_[0])
                        X_encoded[col] = le.transform(vals_mapped)
                    else:
                        # Fallback simple encoding if not fit
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].fillna("missing").astype(str))
        return X_encoded

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBayesClassifier":
        X_clean = self._preprocess(X, is_training=True)
        self.classes_ = sorted(list(y.astype(str).unique()))
        
        # Fit GaussianNB
        self.model.fit(X_clean, y.astype(str))
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Naive Bayes model has not been trained yet")
        X_clean = self._preprocess(X, is_training=False)
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Naive Bayes model has not been trained yet")
        X_clean = self._preprocess(X, is_training=False)
        return self.model.predict_proba(X_clean)
