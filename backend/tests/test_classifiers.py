import pytest
import pandas as pd
import numpy as np
from app.ml.algorithms.id3 import ID3Classifier
from app.ml.algorithms.j48 import J48Classifier
from app.ml.algorithms.naive_bayes import NaiveBayesClassifier
from app.ml.algorithms.knn import KNNClassifier

@pytest.fixture
def dummy_data():
    """
    Generate synthetic classification data.
    outlook: nominal
    temperature: numeric
    play: target (yes/no)
    """
    X = pd.DataFrame({
        "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "overcast", "sunny"],
        "temperature": [85, 80, 83, 70, 68, 64, 72]
    })
    y = pd.Series(["no", "no", "yes", "yes", "yes", "yes", "no"])
    return X, y

def test_classifiers_initialization():
    """
    Ensure all models can be instantiated and adhere to BaseClassifier interface.
    """
    id3 = ID3Classifier()
    j48 = J48Classifier()
    nb = NaiveBayesClassifier()
    knn = KNNClassifier()
    
    assert not id3.is_trained
    assert not j48.is_trained
    assert not nb.is_trained
    assert not knn.is_trained
    
    assert hasattr(id3, "fit") and hasattr(id3, "predict")
    assert hasattr(j48, "fit") and hasattr(j48, "predict")
    assert hasattr(nb, "fit") and hasattr(nb, "predict")
    assert hasattr(knn, "fit") and hasattr(knn, "predict")

def test_id3_fit_predict(dummy_data):
    X, y = dummy_data
    clf = ID3Classifier()
    # ID3 uses nominals only, so it drops or treats temperature as string
    clf.fit(X, y)
    assert clf.is_trained
    
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({"yes", "no"})

def test_j48_fit_predict(dummy_data):
    X, y = dummy_data
    clf = J48Classifier(hyperparameters={"min_instances": 1})
    clf.fit(X, y)
    assert clf.is_trained
    
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({"yes", "no"})

def test_naive_bayes_fit_predict(dummy_data):
    X, y = dummy_data
    clf = NaiveBayesClassifier()
    clf.fit(X, y)
    assert clf.is_trained
    
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({"yes", "no"})

def test_knn_fit_predict(dummy_data):
    X, y = dummy_data
    clf = KNNClassifier(hyperparameters={"n_neighbors": 3})
    clf.fit(X, y)
    assert clf.is_trained
    
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({"yes", "no"})
