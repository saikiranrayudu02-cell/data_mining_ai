import importlib
import time
import os
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from app.ml.algorithms.base import BaseClassifier
from app.ml.evaluator.model_evaluator import ModelEvaluator
from app.schemas.dataset import EvaluationMetrics
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Dynamic Algorithm Mapping
ALGORITHM_REGISTRY = {
    "ID3": ("app.ml.algorithms.id3", "ID3Classifier"),
    "J48": ("app.ml.algorithms.j48", "J48Classifier"),
    "NaiveBayes": ("app.ml.algorithms.naive_bayes", "NaiveBayesClassifier"),
    "KNN": ("app.ml.algorithms.knn", "KNNClassifier")
}

class ModelTrainer:
    """
    Handles dynamic algorithm instantiation, evaluation modes (Cross-validation,
    Percentage split, Training set), metrics computation, and model serialization.
    """
    
    @staticmethod
    def _instantiate_algorithm(algorithm_name: str, hyperparameters: Dict[str, Any]) -> BaseClassifier:
        if algorithm_name not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{algorithm_name}' is not supported. Choose from {list(ALGORITHM_REGISTRY.keys())}")
            
        module_path, class_name = ALGORITHM_REGISTRY[algorithm_name]
        try:
            logger.info(f"Dynamically importing class {class_name} from {module_path}")
            module = importlib.import_module(module_path)
            clf_class = getattr(module, class_name)
            return clf_class(hyperparameters=hyperparameters)
        except Exception as e:
            logger.error(f"Failed to dynamically load algorithm {algorithm_name}: {str(e)}")
            raise RuntimeError(f"Failed to load algorithm class: {str(e)}")

    @staticmethod
    def train_and_evaluate(
        df: pd.DataFrame,
        target_col: str,
        algorithm_name: str,
        evaluation_mode: str = "cross_validation",
        percentage_split: float = 66.0,
        folds: int = 10,
        random_seed: int = 1,
        test_split: float = 0.2, # Fallback legacy parameter
        hyperparameters: Optional[Dict[str, Any]] = None,
        relation_name: str = "dataset"
    ) -> Tuple[BaseClassifier, EvaluationMetrics, str]:
        """
        Train model dynamically, compute WEKA-style evaluation metrics, and serialize artifact.
        """
        hyperparams = hyperparameters or {}
        
        # 1. Instantiate final model for tree extraction
        final_clf = ModelTrainer._instantiate_algorithm(algorithm_name, hyperparams)
        
        # 2. Separate features X and target label y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
        start_time = time.perf_counter()
        
        y_true_outcomes: List[Any] = []
        y_pred_outcomes: List[Any] = []
        y_prob_list: List[np.ndarray] = []
        
        eval_mode_display = ""
        
        # 3. Execute evaluation mode logic
        if evaluation_mode == "training_set":
            eval_mode_display = "evaluate on training data"
            final_clf.fit(X, y)
            y_pred_arr = final_clf.predict(X)
            y_prob_arr = None
            try:
                y_prob_arr = final_clf.predict_proba(X)
            except Exception:
                pass
                
            y_true_outcomes = y.tolist()
            y_pred_outcomes = y_pred_arr.tolist()
            y_prob_outcomes = y_prob_arr
            
        elif evaluation_mode == "percentage_split":
            split_pct = percentage_split if percentage_split > 0 and percentage_split < 100 else 66.0
            eval_mode_display = f"{split_pct:g}% train, {100.0 - split_pct:g}% test split"
            test_ratio = (100.0 - split_pct) / 100.0
            
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_ratio, random_state=random_seed, stratify=y if len(y.unique()) > 1 else None
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_ratio, random_state=random_seed
                )
                
            final_clf.fit(X_tr, y_tr)
            y_pred_arr = final_clf.predict(X_te)
            y_prob_arr = None
            try:
                y_prob_arr = final_clf.predict_proba(X_te)
            except Exception:
                pass
                
            y_true_outcomes = y_te.tolist()
            y_pred_outcomes = y_pred_arr.tolist()
            y_prob_outcomes = y_prob_arr
            
        else: # "cross_validation"
            min_class_samples = int(y.value_counts().min()) if len(y) > 0 else 1
            n_folds = min(folds, len(y), min_class_samples)
            
            if n_folds < 2:
                eval_mode_display = "evaluate on training data (dataset too small for CV)"
                final_clf.fit(X, y)
                y_pred_arr = final_clf.predict(X)
                y_prob_arr = None
                try:
                    y_prob_arr = final_clf.predict_proba(X)
                except Exception:
                    pass
                    
                y_true_outcomes = y.tolist()
                y_pred_outcomes = y_pred_arr.tolist()
                y_prob_outcomes = y_prob_arr
            else:
                eval_mode_display = f"{n_folds}-fold cross-validation"
                try:
                    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
                    splits = list(cv_splitter.split(X, y))
                except ValueError:
                    cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
                    splits = list(cv_splitter.split(X, y))
                    
                for train_idx, test_idx in splits:
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    
                    fold_clf = ModelTrainer._instantiate_algorithm(algorithm_name, hyperparams)
                    fold_clf.fit(X_tr, y_tr)
                    
                    y_p_fold = fold_clf.predict(X_te)
                    y_pr_fold = None
                    try:
                        y_pr_fold = fold_clf.predict_proba(X_te)
                    except Exception:
                        pass
                        
                    y_true_outcomes.extend(y_te.tolist())
                    y_pred_outcomes.extend(y_p_fold.tolist())
                    if y_pr_fold is not None:
                        y_prob_list.append(y_pr_fold)
                        
                final_clf.fit(X, y)
                y_prob_outcomes = np.vstack(y_prob_list) if y_prob_list else None

        end_time = time.perf_counter()
        mem_after = process.memory_info().rss
        
        train_duration_ms = (end_time - start_time) * 1000.0
        memory_delta_mb = max(0.0, (mem_after - mem_before) / (1024.0 * 1024.0))
        
        # 4. Extract WEKA ASCII Tree and Entropy Stats if available
        weka_tree_text = getattr(final_clf, "get_weka_tree_text", lambda: None)()
        entropy_stats = getattr(final_clf, "entropy_stats_", None)
        
        # Reproducibility Metadata
        reproducibility_meta = {
            "relation_name": relation_name,
            "class_attribute": target_col,
            "algorithm": algorithm_name,
            "hyperparameters": hyperparams,
            "evaluation_mode": evaluation_mode,
            "percentage_split": percentage_split,
            "folds": folds,
            "random_seed": random_seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        model_id = str(uuid.uuid4())
        cm_filename = f"{model_id}_cm.png"
        roc_filename = f"{model_id}_roc.png"
        
        cm_path = str(settings.PLOT_DIR / cm_filename)
        roc_path = str(settings.PLOT_DIR / roc_filename)
        
        cm_url = f"/plots/{cm_filename}"
        roc_url = f"/plots/{roc_filename}"
        
        # Build WEKA parameters string for header
        param_parts = []
        for k, v in hyperparams.items():
            param_parts.append(f"-{k} {v}")
        algorithm_params_str = " ".join(param_parts)
            
        # 5. Evaluate WEKA metrics
        metrics = ModelEvaluator.evaluate(
            y_true=y_true_outcomes,
            y_pred=y_pred_outcomes,
            y_prob=y_prob_outcomes,
            labels=final_clf.classes_,
            execution_time_ms=round(train_duration_ms, 2),
            confusion_matrix_path=cm_path,
            roc_curve_path=roc_path,
            confusion_matrix_url=cm_url,
            roc_curve_url=roc_url,
            memory_used_mb=round(memory_delta_mb, 4),
            relation_name=relation_name,
            algorithm_name=algorithm_name,
            algorithm_params_str=algorithm_params_str,
            evaluation_mode_display=eval_mode_display,
            weka_tree_text=weka_tree_text,
            entropy_stats=entropy_stats,
            reproducibility_meta=reproducibility_meta
        )
        
        # 6. Save model using joblib
        model_path = settings.MODEL_DIR / f"{model_id}.joblib"
        logger.info(f"Serializing trained model to {model_path}")
        joblib.dump({"model": final_clf, "metrics": metrics}, model_path)
        
        return final_clf, metrics, model_id
