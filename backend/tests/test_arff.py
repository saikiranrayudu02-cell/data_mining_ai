import pytest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from app.ml.arff_parser import ARFFParser

@pytest.fixture
def sample_arff_file():
    """
    Create a mock weather ARFF file.
    Includes nominal variables, numeric variables, and missing value indicators (?).
    """
    arff_content = """@relation weather

@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute humidity numeric
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}

@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,86,FALSE,yes
rainy,70,96,FALSE,yes
rainy,?,80,FALSE,yes
rainy,65,70,TRUE,no
overcast,64,65,TRUE,yes
sunny,72,95,?,no
sunny,69,70,FALSE,yes
"""
    # Create temporary file inside context
    with tempfile.NamedTemporaryFile(suffix=".arff", mode="w", delete=False) as temp:
        temp.write(arff_content)
        temp_path = Path(temp.name)
    
    yield temp_path
    
    # Cleanup after test
    if temp_path.exists():
        temp_path.unlink()

def test_arff_parser_and_preprocess(sample_arff_file):
    """
    Validate metadata extraction, missing values detection, imputation, encoding, and scaling.
    """
    # 1. Parse ARFF file
    parsed = ARFFParser.parse(sample_arff_file)
    
    assert parsed["relation"] == "weather"
    assert parsed["num_instances"] == 9
    assert parsed["class_attribute"] == "play"
    
    # Check attributes mapping
    attrs = parsed["attributes"]
    assert len(attrs) == 5
    assert attrs[0]["name"] == "outlook"
    assert attrs[0]["type"] == "nominal"
    assert attrs[0]["values"] == ["sunny", "overcast", "rainy"]
    
    assert attrs[1]["name"] == "temperature"
    assert attrs[1]["type"] == "numeric"
    
    # Check detected missing values
    missing = parsed["missing_values"]
    assert missing["temperature"] == 1
    assert missing["windy"] == 1
    assert missing["play"] == 0

    df = parsed["dataframe"]
    assert isinstance(df, pd.DataFrame)
    
    # 2. Impute missing values
    df_imputed = ARFFParser.impute_missing_values(df, attrs)
    assert df_imputed["temperature"].isna().sum() == 0
    # Imputed with mean: (85+80+83+70+65+64+72+69)/8 = 588/8 = 73.5
    assert df_imputed["temperature"].iloc[4] == 73.5
    # Nominal imputed with mode
    assert df_imputed["windy"].isna().sum() == 0
    assert df_imputed["windy"].iloc[7] == "FALSE" # 'FALSE' is mode (5 FALSE vs 3 TRUE)

    # 3. Categorical encoding
    df_encoded, mappings = ARFFParser.encode_categorical(df_imputed, attrs)
    assert df_encoded["outlook"].dtype in [np.int32, np.int64]
    assert set(mappings["outlook"]) == {"sunny", "overcast", "rainy"}
    
    # 4. Numerical normalization
    df_normalized = ARFFParser.normalize_numerical(df_encoded, attrs)
    # Check range is normalized [0, 1]
    assert df_normalized["temperature"].min() >= 0.0
    assert df_normalized["temperature"].max() <= 1.0
