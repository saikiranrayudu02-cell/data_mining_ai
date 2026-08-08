export interface AttributeMetadata {
  name: string;
  type: string;
  values?: string[];
}

export interface DatasetMetadata {
  dataset_id: string;
  relation_name: string;
  num_attributes: number;
  num_instances: number;
  attributes: AttributeMetadata[];
  target_attribute?: string;
  class_attribute?: string;
  class_distribution?: Record<string, number>;
  dataset_hash?: string;
}

export type AlgorithmType = "ID3" | "J48" | "NaiveBayes" | "KNN";

export type EvaluationMode = "cross_validation" | "percentage_split" | "training_set";

export interface ClassifyRequest {
  dataset_id: string;
  algorithm: AlgorithmType;
  target_attribute?: string;
  evaluation_mode?: EvaluationMode;
  percentage_split?: number;
  folds?: number;
  random_seed?: number;
  test_split?: number;
  hyperparameters?: Record<string, unknown>;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface DetailedAccuracyClass {
  class_name: string;
  tp_rate: number;
  fp_rate: number;
  precision: number;
  recall: number;
  f_measure: number;
  mcc: number;
  roc_area: number;
  prc_area: number;
}

export interface EntropyStat {
  attribute_name: string;
  entropy: number;
  info_gain: number;
  split_info?: number;
  gain_ratio?: number;
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: ConfusionMatrix;
  roc_auc?: number;
  cohen_kappa: number;
  classification_report: string;
  confusion_matrix_plot_url?: string;
  roc_curve_plot_url?: string;
  memory_used_mb?: number;
  execution_time_ms: number;

  // WEKA Explorer Metrics
  correctly_classified_instances?: number;
  correctly_classified_pct?: number;
  incorrectly_classified_instances?: number;
  incorrectly_classified_pct?: number;
  mae?: number;
  rmse?: number;
  rae?: number;
  rrse?: number;
  total_instances?: number;
  detailed_accuracy_by_class?: DetailedAccuracyClass[];
  weighted_accuracy?: DetailedAccuracyClass;
  entropy_stats?: EntropyStat[];
  raw_weka_output?: string;
  weka_tree_text?: string;
  reproducibility?: Record<string, unknown>;
}

export interface DecisionTreeNode {
  id: string;
  type?: string;
  label: string;
  position?: { x: number; y: number };
  data?: Record<string, unknown>;
}

export interface DecisionTreeEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface DecisionTreeStructure {
  nodes: DecisionTreeNode[];
  edges: DecisionTreeEdge[];
}

export interface ClassifyResponse {
  model_id: string;
  algorithm: AlgorithmType;
  metrics: EvaluationMetrics;
  rules: string[];
  tree?: DecisionTreeStructure;
  tree_png_url?: string;
  tree_depth?: number;
  tree_leaf_nodes?: number;
  tree_internal_nodes?: number;
}

export interface CompareRequest {
  dataset_id: string;
  target_attribute?: string;
  evaluation_mode?: EvaluationMode;
  percentage_split?: number;
  folds?: number;
  random_seed?: number;
  test_split?: number;
  hyperparameters?: Record<string, Record<string, unknown>>;
}

export interface CompareResponse {
  dataset_id: string;
  results: Record<AlgorithmType, EvaluationMetrics>;
  best_algorithm?: string;
  worst_algorithm?: string;
  rankings?: string[];
  ranking_reason?: string;
}

export interface ExportRequest {
  model_id?: string;
  dataset_id: string;
  format: "pdf" | "html" | "json";
  include_metrics: boolean;
  include_rules: boolean;
  include_tree: boolean;
}

export interface ExportResponse {
  download_url: string;
  format: string;
  filename: string;
}

export interface DatasetPreviewResponse {
  columns: string[];
  raw_data: Record<string, unknown>[];
  processed_data: Record<string, unknown>[];
}
