from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from functools import lru_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ARFFParser:
    """
    Parses ARFF (Attribute-Relation File Format) files using scipy.io.arff.
    Performs data cleaning, missing value extraction, imputation, and normalization.
    """

    @staticmethod
    def parse(file_path: Path) -> Dict[str, Any]:
        """
        Read ARFF file, fetching cached output if available.
        """
        return ARFFParser._parse_cached(str(file_path))

    @staticmethod
    @lru_cache(maxsize=16)
    def _parse_cached(file_path_str: str) -> Dict[str, Any]:
        logger.info(f"Loading and parsing ARFF file using Scipy (cached lookup): {file_path_str}")
        try:
            # Parse with scipy
            data, meta = arff.loadarff(file_path_str)
            df = pd.DataFrame(data)
            
            relation = str(meta.name) if hasattr(meta, 'name') else "unknown"
            attributes_names = meta.names()
            
            # Format and decode elements (scipy reads nominal variables as bytes)
            parsed_attributes = []
            for name in attributes_names:
                attr_meta = meta[name]
                attr_type = attr_meta[0] # 'numeric', 'nominal', 'string', 'date'
                
                # Check nominal classes
                nominal_values = None
                if attr_type == 'nominal':
                    nominal_values = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in attr_meta[1]]
                
                parsed_attributes.append({
                    "name": name,
                    "type": "nominal" if attr_type == 'nominal' else "numeric",
                    "values": nominal_values
                })

            # Decode bytes columns inside dataframe
            for col in df.columns:
                if df[col].dtype == object or isinstance(df[col].iloc[0], bytes):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    
            # Standardize missing values indicator '?' or None inside the DataFrame
            # For objects/categorical columns, replace '?' with nan
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) or df[col].dtype.kind in 'OSU':
                    df[col] = df[col].astype(object).replace('?', np.nan)
            
            # Calculate instance metrics
            num_instances = len(df)
            
            # Detect missing values counts per attribute
            missing_counts = {}
            for col in df.columns:
                missing_counts[col] = int(df[col].isna().sum())
                
            # Compute MD5 file hash for experiment reproducibility
            import hashlib
            with open(file_path_str, "rb") as f_hash:
                dataset_hash = hashlib.md5(f_hash.read()).hexdigest()

            # Class attribute is usually the last attribute by default in WEKA
            class_attribute = attributes_names[-1] if len(attributes_names) > 0 else None
            
            # Compute class distribution
            class_distribution = {}
            if class_attribute and class_attribute in df.columns:
                val_counts = df[class_attribute].value_counts()
                class_distribution = {str(k): int(v) for k, v in val_counts.items()}

            return {
                "relation": relation,
                "attributes": parsed_attributes,
                "dataframe": df,
                "num_instances": num_instances,
                "class_attribute": class_attribute,
                "class_distribution": class_distribution,
                "missing_values": missing_counts,
                "dataset_hash": dataset_hash
            }
        except Exception as e:
            logger.error(f"Failed to process ARFF dataset: {str(e)}")
            raise ValueError(f"Failed to parse ARFF dataset file: {str(e)}")

    @staticmethod
    def impute_missing_values(df: pd.DataFrame, attributes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Handle missing values in the dataframe:
        - Imputes numerical attributes with mean value.
        - Imputes nominal attributes with mode value.
        """
        df_imputed = df.copy()
        for attr in attributes:
            col = attr["name"]
            if df_imputed[col].isna().sum() > 0:
                if attr["type"] == "numeric":
                    mean_val = df_imputed[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0.0
                    df_imputed[col] = df_imputed[col].fillna(mean_val)
                    logger.info(f"Imputed missing numeric column {col} with mean value {mean_val}")
                else:
                    mode_series = df_imputed[col].mode()
                    mode_val = mode_series[0] if not mode_series.empty else "unknown"
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                    logger.info(f"Imputed missing nominal column {col} with mode value {mode_val}")
        return df_imputed

    @staticmethod
    def encode_categorical(df: pd.DataFrame, attributes: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Encode nominal/categorical attributes using scikit-learn LabelEncoder.
        Returns the encoded dataframe and the mapping details.
        """
        df_encoded = df.copy()
        mappings = {}
        for attr in attributes:
            col = attr["name"]
            if attr["type"] == "nominal":
                le = LabelEncoder()
                # Ensure missing values are imputed before encoding or treat na as string
                vals = df_encoded[col].fillna("missing_value").astype(str)
                df_encoded[col] = le.fit_transform(vals)
                mappings[col] = [str(c) for c in le.classes_]
                logger.info(f"Encoded nominal column {col} with label classes {mappings[col]}")
        return df_encoded, mappings

    @staticmethod
    def normalize_numerical(df: pd.DataFrame, attributes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize numerical values to range [0, 1] using MinMaxScaler.
        """
        df_normalized = df.copy()
        numeric_cols = [attr["name"] for attr in attributes if attr["type"] == "numeric"]
        
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols].fillna(0.0))
            logger.info(f"Normalized numerical columns: {numeric_cols}")
            
        return df_normalized
