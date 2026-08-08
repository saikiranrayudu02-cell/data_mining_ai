import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from app.ml.arff_parser import ARFFParser
from app.ml.trainer.model_trainer import ModelTrainer
from app.ml.algorithms.j48 import J48Classifier
from app.ml.algorithms.id3 import ID3Classifier
from app.ml.algorithms.naive_bayes import NaiveBayesClassifier
from app.ml.algorithms.knn import KNNClassifier

IRIS_PATH = Path("/Users/maggi/data_mine_pro/datasets/iris.arff")

def test_arff_parsing_iris():
    assert IRIS_PATH.exists(), "iris.arff dataset must exist"
    parsed = ARFFParser.parse(IRIS_PATH)
    
    assert parsed["relation"].lower() == "iris"
    assert len(parsed["attributes"]) == 5
    assert parsed["num_instances"] == 150
    assert parsed["class_attribute"] == "class"
    assert parsed["class_distribution"] == {
        "Iris-setosa": 50,
        "Iris-versicolor": 50,
        "Iris-virginica": 50
    }
    assert parsed["dataset_hash"] is not None and len(parsed["dataset_hash"]) > 0

def test_j48_weka_cross_validation():
    parsed = ARFFParser.parse(IRIS_PATH)
    df = parsed["dataframe"]
    
    clf, metrics, model_id = ModelTrainer.train_and_evaluate(
        df=df,
        target_col="class",
        algorithm_name="J48",
        evaluation_mode="cross_validation",
        folds=10,
        random_seed=1,
        relation_name="iris"
    )
    
    assert isinstance(clf, J48Classifier)
    assert metrics.accuracy >= 0.90
    assert metrics.cohen_kappa >= 0.85
    assert metrics.total_instances == 150
    assert metrics.correctly_classified_instances + metrics.incorrectly_classified_instances == 150
    assert len(metrics.detailed_accuracy_by_class) == 3
    assert metrics.weighted_accuracy is not None
    assert "=== Summary ===" in metrics.raw_weka_output
    assert "=== Confusion Matrix ===" in metrics.raw_weka_output
    assert "J48 pruned tree" in metrics.raw_weka_output

def test_id3_weka_cross_validation():
    parsed = ARFFParser.parse(IRIS_PATH)
    df = parsed["dataframe"]
    
    clf, metrics, model_id = ModelTrainer.train_and_evaluate(
        df=df,
        target_col="class",
        algorithm_name="ID3",
        evaluation_mode="cross_validation",
        folds=10,
        random_seed=1,
        relation_name="iris"
    )
    
    assert isinstance(clf, ID3Classifier)
    assert metrics.accuracy >= 0.80
    assert metrics.cohen_kappa >= 0.70
    assert metrics.total_instances == 150
    assert len(metrics.entropy_stats) > 0
    assert "=== Detailed Accuracy By Class ===" in metrics.raw_weka_output

def test_naive_bayes_weka_cross_validation():
    parsed = ARFFParser.parse(IRIS_PATH)
    df = parsed["dataframe"]
    
    clf, metrics, model_id = ModelTrainer.train_and_evaluate(
        df=df,
        target_col="class",
        algorithm_name="NaiveBayes",
        evaluation_mode="cross_validation",
        folds=10,
        random_seed=1,
        relation_name="iris"
    )
    
    assert isinstance(clf, NaiveBayesClassifier)
    assert metrics.accuracy >= 0.90
    assert metrics.cohen_kappa >= 0.85

def test_knn_weka_cross_validation():
    parsed = ARFFParser.parse(IRIS_PATH)
    df = parsed["dataframe"]
    
    clf, metrics, model_id = ModelTrainer.train_and_evaluate(
        df=df,
        target_col="class",
        algorithm_name="KNN",
        evaluation_mode="cross_validation",
        folds=10,
        random_seed=1,
        hyperparameters={"n_neighbors": 1},
        relation_name="iris"
    )
    
    assert isinstance(clf, KNNClassifier)
    assert metrics.accuracy >= 0.90
    assert metrics.cohen_kappa >= 0.85
