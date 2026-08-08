from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Dataset Engine Schemas ---
class AttributeMetadata(BaseModel):
    name: str = Field(..., description="Name of the attribute")
    type: str = Field(..., description="Data type (numeric, nominal)")
    values: Optional[List[str]] = Field(None, description="Nominal categories if nominal type")

class DatasetInfoResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier")
    relation_name: str = Field(..., description="ARFF relation name")
    num_attributes: int = Field(..., description="Total attributes count")
    num_instances: int = Field(..., description="Total rows/records count")
    attributes: List[AttributeMetadata] = Field(..., description="Dataset schema attributes")
    class_attribute: Optional[str] = Field(None, description="Class target field")
    class_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution count per class label")
    missing_values_count: Dict[str, int] = Field(..., description="Count of missing values per attribute")
    dataset_hash: Optional[str] = Field(None, description="MD5 hash of dataset content for reproducibility")

class DatasetUploadResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier")
    relation_name: str = Field(..., description="ARFF relation name")
    num_attributes: int = Field(..., description="Total attributes count")
    num_instances: int = Field(..., description="Total instances count")
    message: str = Field("Dataset uploaded and processed successfully", description="Status message")

class DatasetPreviewResponse(BaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    columns: List[str] = Field(..., description="Ordered column names")
    raw_data: List[Dict[str, Any]] = Field(..., description="Raw records lists (first N records)")
    processed_data: List[Dict[str, Any]] = Field(..., description="Preprocessed records lists (first N records)")


# --- Classification & Visualization Schemas ---
class ClassifyRequest(BaseModel):
    dataset_id: Optional[str] = Field(None, description="Target dataset identifier")
    algorithm: str = Field(..., description="Classification algorithm (ID3, J48, NaiveBayes, KNN)")
    target_attribute: Optional[str] = Field(None, description="The class attribute to classify (defaults to final attribute)")
    evaluation_mode: str = Field("cross_validation", description="Evaluation mode: training_set, percentage_split, cross_validation")
    percentage_split: float = Field(66.0, description="Train percentage for percentage_split mode (default 66%)")
    folds: int = Field(10, description="Number of folds for cross_validation mode (default 10)")
    random_seed: int = Field(1, description="Random seed for reproducibility")
    test_split: float = Field(0.2, description="Validation split size (0.0 to 1.0) fallback")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyper-parameters")

class ConfusionMatrix(BaseModel):
    labels: List[str] = Field(..., description="Class labels index")
    matrix: List[List[int]] = Field(..., description="Matrix rows")

class DetailedAccuracyClass(BaseModel):
    class_name: str = Field(..., description="Class label name")
    tp_rate: float = Field(0.0, description="True Positive Rate")
    fp_rate: float = Field(0.0, description="False Positive Rate")
    precision: float = Field(0.0, description="Precision")
    recall: float = Field(0.0, description="Recall")
    f_measure: float = Field(0.0, description="F-Measure")
    mcc: float = Field(0.0, description="Matthews Correlation Coefficient")
    roc_area: float = Field(0.0, description="Area Under ROC Curve")
    prc_area: float = Field(0.0, description="Area Under Precision-Recall Curve")

class EntropyStat(BaseModel):
    attribute_name: str = Field(..., description="Candidate split attribute name")
    entropy: float = Field(..., description="Entropy of feature split")
    info_gain: float = Field(..., description="Information Gain")
    split_info: Optional[float] = Field(None, description="Split Information (C4.5/J48)")
    gain_ratio: Optional[float] = Field(None, description="Gain Ratio (C4.5/J48)")

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrix
    roc_auc: Optional[float] = None
    cohen_kappa: float = Field(..., description="Cohen's Kappa statistic score")
    classification_report: str = Field(..., description="Detailed textual classification report")
    confusion_matrix_plot_url: Optional[str] = Field(None, description="Static URL of the confusion matrix heat map")
    roc_curve_plot_url: Optional[str] = Field(None, description="Static URL of the ROC curve graph plot")
    memory_used_mb: float = Field(0.0, description="Memory space delta consumed during fitting (MB)")
    execution_time_ms: float

    # --- WEKA-Style Explorer Metrics ---
    correctly_classified_instances: int = Field(0, description="Count of correctly predicted instances")
    correctly_classified_pct: float = Field(0.0, description="Percentage of correctly predicted instances")
    incorrectly_classified_instances: int = Field(0, description="Count of incorrectly predicted instances")
    incorrectly_classified_pct: float = Field(0.0, description="Percentage of incorrectly predicted instances")
    mae: float = Field(0.0, description="Mean Absolute Error")
    rmse: float = Field(0.0, description="Root Mean Squared Error")
    rae: float = Field(0.0, description="Relative Absolute Error (%)")
    rrse: float = Field(0.0, description="Root Relative Squared Error (%)")
    total_instances: int = Field(0, description="Total evaluated test instances count")
    
    detailed_accuracy_by_class: List[DetailedAccuracyClass] = Field(default_factory=list, description="Per-class detailed metrics")
    weighted_accuracy: Optional[DetailedAccuracyClass] = Field(None, description="Weighted average metrics across all classes")
    entropy_stats: List[EntropyStat] = Field(default_factory=list, description="Candidate split attributes Entropy and Gain stats")
    raw_weka_output: str = Field("", description="Complete ASCII formatted WEKA Explorer result report text")
    weka_tree_text: Optional[str] = Field(None, description="ASCII indented WEKA-style tree representation")
    reproducibility: Dict[str, Any] = Field(default_factory=dict, description="Experiment metadata (hash, seed, params, timestamp)")

# React Flow node representation
class DecisionTreeNode(BaseModel):
    id: str
    type: str = "default"  # 'default', 'input', 'output'
    label: str
    data: Dict[str, Any] = Field(default_factory=dict)
    
class DecisionTreeEdge(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None  # Branch condition text

class DecisionTreeStructure(BaseModel):
    nodes: List[DecisionTreeNode]
    edges: List[DecisionTreeEdge]

class ClassifyResponse(BaseModel):
    model_id: str
    algorithm: str
    metrics: EvaluationMetrics
    rules: List[str] = Field(default_factory=list, description="IF-THEN rule extractions")
    tree: Optional[DecisionTreeStructure] = Field(None, description="Decision tree nodes and edges for React Flow")
    tree_png_url: Optional[str] = Field(None, description="Static URL of the compiled decision tree PNG plot")
    tree_depth: Optional[int] = Field(None, description="Maximum path depth of the decision tree")
    tree_leaf_nodes: Optional[int] = Field(None, description="Total count of leaf nodes in the tree")
    tree_internal_nodes: Optional[int] = Field(None, description="Total count of internal split nodes in the tree")

class CompareRequest(BaseModel):
    dataset_id: str
    target_attribute: Optional[str] = None
    evaluation_mode: str = Field("cross_validation", description="Evaluation mode: training_set, percentage_split, cross_validation")
    percentage_split: float = Field(66.0, description="Train percentage for percentage_split mode (default 66%)")
    folds: int = Field(10, description="Number of folds for cross_validation mode (default 10)")
    random_seed: int = Field(1, description="Random seed for reproducibility")
    hyperparameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Algorithm-specific hyper-parameters")

class CompareResponse(BaseModel):
    dataset_id: str
    results: Dict[str, EvaluationMetrics] = Field(..., description="Comparison mappings by algorithm name")
    best_algorithm: Optional[str] = Field(None, description="The top performing algorithm name based on multi-metric score")
    worst_algorithm: Optional[str] = Field(None, description="The lowest performing algorithm name based on multi-metric score")
    rankings: List[str] = Field(default_factory=list, description="Sorted list of algorithms from best to worst")
    ranking_reason: str = Field("", description="Transparent multi-metric rationale for algorithm rankings")

class ExportRequest(BaseModel):
    model_id: Optional[str] = None
    dataset_id: str
    format: str = Field("pdf", description="Export format (pdf, html, json)")
    include_metrics: bool = True
    include_rules: bool = True
    include_tree: bool = True

class ExportResponse(BaseModel):
    download_url: str
    format: str
    filename: str
