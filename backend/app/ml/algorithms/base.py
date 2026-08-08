from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

class BaseClassifier(ABC):
    """
    Abstract Base Class for DataMine AI classifiers.
    Standardizes models interfaces for dynamic loading.
    """
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        self.hyperparameters = hyperparameters or {}
        self.is_trained: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier":
        """
        Train the classification model.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target classes for input features X.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class membership probabilities for input features X.
        """
        pass
