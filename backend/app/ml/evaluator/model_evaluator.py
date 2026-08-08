import matplotlib
matplotlib.use('Agg') # Force non-GUI background backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_auc_score, 
    cohen_kappa_score, 
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
    matthews_corrcoef
)
from app.schemas.dataset import EvaluationMetrics, ConfusionMatrix, DetailedAccuracyClass, EntropyStat
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    WEKA Explorer-compatible Model Evaluator:
    Computes WEKA summary metrics (Accuracy, MAE, RMSE, RAE, RRSE, Kappa),
    Detailed Accuracy By Class (TP Rate, FP Rate, Precision, Recall, F-Measure, MCC, ROC Area, PRC Area),
    One-vs-Rest multiclass ROC curves, and formats raw WEKA Explorer multiline ASCII text reports.
    """
    
    @staticmethod
    def evaluate(
        y_true: List[Any],
        y_pred: List[Any],
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        execution_time_ms: float = 0.0,
        confusion_matrix_path: Optional[str] = None,
        roc_curve_path: Optional[str] = None,
        confusion_matrix_url: Optional[str] = None,
        roc_curve_url: Optional[str] = None,
        memory_used_mb: float = 0.0,
        relation_name: str = "dataset",
        algorithm_name: str = "Classifier",
        algorithm_params_str: str = "",
        evaluation_mode_display: str = "10-fold cross-validation",
        weka_tree_text: Optional[str] = None,
        entropy_stats: Optional[List[Dict[str, Any]]] = None,
        reproducibility_meta: Optional[Dict[str, Any]] = None
    ) -> EvaluationMetrics:
        """
        Compute standard validation metrics, per-class WEKA statistics, MAE/RMSE/RAE/RRSE,
        and generate plot graphics and ASCII text reports.
        """
        y_t = np.array([str(x) for x in y_true])
        y_p = np.array([str(x) for x in y_pred])
        
        if labels is None:
            labels = sorted(list(set(y_t)))
        labels = [str(l) for l in labels]
        
        total_instances = len(y_t)
        if total_instances == 0:
            total_instances = 1
            
        correct_count = int((y_t == y_p).sum())
        incorrect_count = total_instances - correct_count
        correct_pct = (correct_count / total_instances) * 100.0
        incorrect_pct = (incorrect_count / total_instances) * 100.0
        
        acc = float(accuracy_score(y_t, y_p))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="weighted", zero_division=0
        )
        kappa = float(cohen_kappa_score(y_t, y_p))
        report = str(classification_report(y_t, y_p, labels=labels, zero_division=0))
        
        cm = confusion_matrix(y_t, y_p, labels=labels)
        conf_matrix = ConfusionMatrix(
            labels=labels,
            matrix=cm.tolist()
        )
        
        # --- Calculate MAE, RMSE, RAE, RRSE ---
        num_classes = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        
        # Target one-hot matrix Y_true (N x C)
        Y_true = np.zeros((total_instances, num_classes))
        for i, val in enumerate(y_t):
            if val in label_to_idx:
                Y_true[i, label_to_idx[val]] = 1.0
                
        # Predicted probability matrix P_pred (N x C)
        if y_prob is not None and y_prob.shape == (total_instances, num_classes):
            P_pred = y_prob
        else:
            P_pred = np.zeros((total_instances, num_classes))
            for i, val in enumerate(y_p):
                if val in label_to_idx:
                    P_pred[i, label_to_idx[val]] = 1.0
                    
        # Class prior distribution
        priors = Y_true.mean(axis=0)
        
        mae = float(np.mean(np.abs(Y_true - P_pred)))
        rmse = float(np.sqrt(np.mean((Y_true - P_pred) ** 2)))
        
        # Relative errors against prior baseline
        prior_abs_diff = np.mean(np.abs(Y_true - priors))
        prior_sq_diff = np.mean((Y_true - priors) ** 2)
        
        rae = float((mae / prior_abs_diff * 100.0)) if prior_abs_diff > 0 else 0.0
        rrse = float((rmse / np.sqrt(prior_sq_diff) * 100.0)) if prior_sq_diff > 0 else 0.0
        
        # --- Calculate Detailed Accuracy By Class ---
        detailed_by_class: List[DetailedAccuracyClass] = []
        
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        tot_tn = 0
        
        w_tp_rate = 0.0
        w_fp_rate = 0.0
        w_prec = 0.0
        w_rec = 0.0
        w_f1 = 0.0
        w_mcc = 0.0
        w_roc_auc = 0.0
        w_prc_auc = 0.0
        
        all_roc_aucs = []
        
        for c_idx, c_label in enumerate(labels):
            binary_true = (y_t == c_label).astype(int)
            binary_pred = (y_p == c_label).astype(int)
            
            tp = int(((binary_true == 1) & (binary_pred == 1)).sum())
            fp = int(((binary_true == 0) & (binary_pred == 1)).sum())
            fn = int(((binary_true == 1) & (binary_pred == 0)).sum())
            tn = int(((binary_true == 0) & (binary_pred == 0)).sum())
            
            support = int(binary_true.sum())
            weight = support / total_instances
            
            tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp_rate
            f_meas = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            
            # Matthews Correlation Coefficient
            try:
                mcc_val = float(matthews_corrcoef(binary_true, binary_pred))
            except Exception:
                mcc_val = 0.0
                
            # ROC AUC per class
            roc_auc_val = 0.5
            try:
                if len(set(binary_true)) > 1:
                    probs_c = P_pred[:, c_idx]
                    roc_auc_val = float(roc_auc_score(binary_true, probs_c))
            except Exception:
                roc_auc_val = 0.5
            all_roc_aucs.append(roc_auc_val)
            
            # PRC AUC per class
            prc_auc_val = 0.5
            try:
                if len(set(binary_true)) > 1:
                    probs_c = P_pred[:, c_idx]
                    prec_arr, rec_arr, _ = precision_recall_curve(binary_true, probs_c)
                    prc_auc_val = float(auc(rec_arr, prec_arr))
            except Exception:
                prc_auc_val = 0.5
                
            cls_stat = DetailedAccuracyClass(
                class_name=c_label,
                tp_rate=round(tp_rate, 3),
                fp_rate=round(fp_rate, 3),
                precision=round(p, 3),
                recall=round(r, 3),
                f_measure=round(f_meas, 3),
                mcc=round(mcc_val, 3),
                roc_area=round(roc_auc_val, 3),
                prc_area=round(prc_auc_val, 3)
            )
            detailed_by_class.append(cls_stat)
            
            w_tp_rate += weight * tp_rate
            w_fp_rate += weight * fp_rate
            w_prec += weight * p
            w_rec += weight * r
            w_f1 += weight * f_meas
            w_mcc += weight * mcc_val
            w_roc_auc += weight * roc_auc_val
            w_prc_auc += weight * prc_auc_val
            
        weighted_avg = DetailedAccuracyClass(
            class_name="Weighted Avg.",
            tp_rate=round(w_tp_rate, 3),
            fp_rate=round(w_fp_rate, 3),
            precision=round(w_prec, 3),
            recall=round(w_rec, 3),
            f_measure=round(w_f1, 3),
            mcc=round(w_mcc, 3),
            roc_area=round(w_roc_auc, 3),
            prc_area=round(w_prc_auc, 3)
        )
        
        overall_roc_auc = float(np.mean(all_roc_aucs)) if all_roc_aucs else 0.5
        
        # --- Generate WEKA Raw Text Report ---
        raw_weka_text = ModelEvaluator._generate_weka_raw_text(
            relation_name=relation_name,
            algorithm_name=algorithm_name,
            algorithm_params_str=algorithm_params_str,
            evaluation_mode_display=evaluation_mode_display,
            total_instances=total_instances,
            correct_count=correct_count,
            correct_pct=correct_pct,
            incorrect_count=incorrect_count,
            incorrect_pct=incorrect_pct,
            kappa=kappa,
            mae=mae,
            rmse=rmse,
            rae=rae,
            rrse=rrse,
            detailed_by_class=detailed_by_class,
            weighted_avg=weighted_avg,
            cm=cm,
            labels=labels,
            weka_tree_text=weka_tree_text,
            execution_time_ms=execution_time_ms
        )
        
        # --- Generate Plots ---
        if confusion_matrix_path:
            try:
                ModelEvaluator._plot_confusion_matrix(cm, labels, confusion_matrix_path)
            except Exception as e:
                logger.error(f"Failed to plot Confusion Matrix: {str(e)}")
                
        if roc_curve_path:
            try:
                ModelEvaluator._plot_multiclass_roc(Y_true, P_pred, labels, roc_curve_path)
            except Exception as e:
                logger.error(f"Failed to plot ROC Curve: {str(e)}")
                
        # Entropy stats Pydantic list
        pydantic_entropy_stats = []
        if entropy_stats:
            for s in entropy_stats:
                pydantic_entropy_stats.append(EntropyStat(
                    attribute_name=s["attribute_name"],
                    entropy=s["entropy"],
                    info_gain=s["info_gain"],
                    split_info=s.get("split_info"),
                    gain_ratio=s.get("gain_ratio")
                ))

        return EvaluationMetrics(
            accuracy=acc,
            precision=float(prec),
            recall=float(rec),
            f1_score=float(f1),
            confusion_matrix=conf_matrix,
            roc_auc=overall_roc_auc,
            cohen_kappa=kappa,
            classification_report=report,
            confusion_matrix_plot_url=confusion_matrix_url,
            roc_curve_plot_url=roc_curve_url,
            memory_used_mb=memory_used_mb,
            execution_time_ms=execution_time_ms,
            correctly_classified_instances=correct_count,
            correctly_classified_pct=round(correct_pct, 4),
            incorrectly_classified_instances=incorrect_count,
            incorrectly_classified_pct=round(incorrect_pct, 4),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            rae=round(rae, 4),
            rrse=round(rrse, 4),
            total_instances=total_instances,
            detailed_accuracy_by_class=detailed_by_class,
            weighted_accuracy=weighted_avg,
            entropy_stats=pydantic_entropy_stats,
            raw_weka_output=raw_weka_text,
            weka_tree_text=weka_tree_text,
            reproducibility=reproducibility_meta or {}
        )

    @staticmethod
    def _generate_weka_raw_text(
        relation_name: str,
        algorithm_name: str,
        algorithm_params_str: str,
        evaluation_mode_display: str,
        total_instances: int,
        correct_count: int,
        correct_pct: float,
        incorrect_count: int,
        incorrect_pct: float,
        kappa: float,
        mae: float,
        rmse: float,
        rae: float,
        rrse: float,
        detailed_by_class: List[DetailedAccuracyClass],
        weighted_avg: DetailedAccuracyClass,
        cm: np.ndarray,
        labels: List[str],
        weka_tree_text: Optional[str],
        execution_time_ms: float
    ) -> str:
        """
        Format an exact ASCII multiline string mimicking WEKA Explorer output.
        """
        lines = []
        lines.append("=== Run information ===")
        lines.append("")
        lines.append(f"Scheme:       weka.classifiers.{algorithm_name} {algorithm_params_str}".strip())
        lines.append(f"Relation:     {relation_name}")
        lines.append(f"Instances:    {total_instances}")
        lines.append(f"Test mode:    {evaluation_mode_display}")
        lines.append("")
        
        if weka_tree_text:
            lines.append("=== Classifier model (full training set) ===")
            lines.append("")
            lines.append(weka_tree_text)
            lines.append("")
            lines.append(f"Time taken to build model: {execution_time_ms / 1000.0:.2f} seconds")
            lines.append("")
            
        lines.append(f"=== Evaluation ({evaluation_mode_display}) ===")
        lines.append("=== Summary ===")
        lines.append("")
        lines.append(f"Correctly Classified Instances         {correct_count:8d}               {correct_pct:7.4f} %")
        lines.append(f"Incorrectly Classified Instances       {incorrect_count:8d}               {incorrect_pct:7.4f} %")
        lines.append(f"Kappa statistic                          {kappa:7.4f}")
        lines.append(f"Mean absolute error                      {mae:7.4f}")
        lines.append(f"Root mean squared error                  {rmse:7.4f}")
        lines.append(f"Relative absolute error                  {rae:7.4f} %")
        lines.append(f"Root relative squared error             {rrse:7.4f} %")
        lines.append(f"Total Number of Instances              {total_instances:8d}")
        lines.append("")
        lines.append("=== Detailed Accuracy By Class ===")
        lines.append("")
        lines.append("                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class")
        
        for cls in detailed_by_class:
            lines.append(
                f"                 {cls.tp_rate:7.3f}  {cls.fp_rate:7.3f}  {cls.precision:9.3f}  {cls.recall:7.3f}  {cls.f_measure:9.3f}  {cls.mcc:7.3f}  {cls.roc_area:8.3f}  {cls.prc_area:8.3f}  {cls.class_name}"
            )
            
        lines.append(
            f"Weighted Avg.   {weighted_avg.tp_rate:7.3f}  {weighted_avg.fp_rate:7.3f}  {weighted_avg.precision:9.3f}  {weighted_avg.recall:7.3f}  {weighted_avg.f_measure:9.3f}  {weighted_avg.mcc:7.3f}  {weighted_avg.roc_area:8.3f}  {weighted_avg.prc_area:8.3f}"
        )
        lines.append("")
        lines.append("=== Confusion Matrix ===")
        lines.append("")
        
        # Build WEKA confusion matrix letters (a, b, c...)
        char_labels = [chr(97 + i) for i in range(len(labels))]
        header_letters = " ".join([f"{cl:3s}" for cl in char_labels])
        lines.append(f"   {header_letters}   <-- classified as")
        
        for i, row in enumerate(cm):
            row_str = " ".join([f"{val:3d}" for val in row])
            lines.append(f" {row_str} |  {char_labels[i]} = {labels[i]}")
            
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: str):
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
        plt.title('Confusion Matrix Heatmap', fontsize=14, pad=15)
        plt.colorbar()
        
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=12, weight='semibold')
                         
        plt.ylabel('True Class label', fontsize=11)
        plt.xlabel('Predicted Class label', fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def _plot_multiclass_roc(Y_true: np.ndarray, P_pred: np.ndarray, labels: List[str], output_path: str):
        plt.figure(figsize=(6, 5))
        colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4']
        
        for c_idx, label in enumerate(labels):
            y_b = Y_true[:, c_idx]
            p_b = P_pred[:, c_idx]
            if len(set(y_b)) > 1:
                fpr, tpr, _ = roc_curve(y_b, p_b)
                roc_auc_val = auc(fpr, tpr)
                color = colors[c_idx % len(colors)]
                plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc_val:.3f})')
                
        plt.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=11)
        plt.ylabel('True Positive Rate (TPR)', fontsize=11)
        plt.title('Multiclass One-vs-Rest ROC Curves', fontsize=13, pad=15)
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
