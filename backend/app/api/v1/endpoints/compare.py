from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas.dataset import CompareRequest, CompareResponse, EvaluationMetrics
from app.ml.arff_parser import ARFFParser
from app.ml.trainer.model_trainer import ModelTrainer
from app.config import settings
import io
import csv
import os
from typing import Tuple, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

def _run_comparison_benchmark(
    dataset_id: str,
    target_col: Optional[str],
    evaluation_mode: str,
    percentage_split: float,
    folds: int,
    random_seed: int,
    hyperparameters: dict
) -> Tuple[dict, str]:
    """
    Helper executing training on all four classifier engines sequentially.
    """
    file_path = settings.UPLOAD_DIR / f"{dataset_id}.arff"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found. Please upload the dataset first."
        )
        
    try:
        parsed = ARFFParser.parse(file_path)
        df = parsed["dataframe"]
        target = target_col or parsed["class_attribute"]
        
        if not target or target not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target class attribute '{target}' not found in dataset."
            )
            
        df_clean = ARFFParser.impute_missing_values(df, parsed["attributes"])
        
        algorithms = ["ID3", "J48", "NaiveBayes", "KNN"]
        results = {}
        
        for alg in algorithms:
            alg_params = hyperparameters.get(alg, {})
            _, metrics, _ = ModelTrainer.train_and_evaluate(
                df=df_clean,
                target_col=target,
                algorithm_name=alg,
                evaluation_mode=evaluation_mode,
                percentage_split=percentage_split,
                folds=folds,
                random_seed=random_seed,
                hyperparameters=alg_params,
                relation_name=parsed.get("relation", "dataset")
            )
            results[alg] = metrics
            
        return results, target
    except Exception as e:
        logger.error(f"Failed benchmark run: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark matrix run failure: {str(e)}"
        )

@router.post("/", response_model=CompareResponse, status_code=status.HTTP_200_OK)
async def compare_classifiers(request: CompareRequest):
    """
    Train and evaluate ID3, J48, Naive Bayes, and KNN algorithms side-by-side.
    Returns complete metrics, best/worst algorithm labels, transparent multi-metric rankings, and rationale.
    """
    if not request.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required"
        )
        
    results, resolved_target = _run_comparison_benchmark(
        dataset_id=request.dataset_id,
        target_col=request.target_attribute,
        evaluation_mode=request.evaluation_mode,
        percentage_split=request.percentage_split,
        folds=request.folds,
        random_seed=request.random_seed,
        hyperparameters=request.hyperparameters
    )
    
    # Calculate transparent multi-metric composite score:
    # Score = 0.40 * Accuracy + 0.30 * F1 + 0.20 * Kappa + 0.10 * AUC
    def _composite_score(alg_name: str) -> float:
        m = results[alg_name]
        acc = m.accuracy
        f1 = m.f1_score
        kappa = max(0.0, m.cohen_kappa)
        auc_val = m.roc_auc if m.roc_auc is not None else 0.5
        return (0.40 * acc) + (0.30 * f1) + (0.20 * kappa) + (0.10 * auc_val)

    sorted_algs = sorted(results.keys(), key=_composite_score, reverse=True)
    best_alg = sorted_algs[0] if sorted_algs else None
    worst_alg = sorted_algs[-1] if sorted_algs else None
    
    ranking_reason = ""
    if best_alg and best_alg in results:
        bm = results[best_alg]
        ranking_reason = (
            f"Ranked based on multi-metric composite score (40% Accuracy, 30% F1-Score, 20% Kappa, 10% AUC). "
            f"Top performer '{best_alg}' achieved Accuracy: {bm.accuracy*100:.2f}%, F1: {bm.f1_score*100:.2f}%, "
            f"Kappa: {bm.cohen_kappa:.4f}, AUC: {bm.roc_auc if bm.roc_auc is not None else 0.5:.3f}."
        )
    
    return CompareResponse(
        dataset_id=request.dataset_id,
        results=results,
        best_algorithm=best_alg,
        worst_algorithm=worst_alg,
        rankings=sorted_algs,
        ranking_reason=ranking_reason
    )

@router.post("/export-csv")
async def export_comparison_csv(request: CompareRequest):
    """
    Run comparison benchmark and stream results as a downloadable CSV attachment.
    """
    if not request.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required"
        )
        
    results, _ = _run_comparison_benchmark(
        dataset_id=request.dataset_id,
        target_col=request.target_attribute,
        evaluation_mode=request.evaluation_mode,
        percentage_split=request.percentage_split,
        folds=request.folds,
        random_seed=request.random_seed,
        hyperparameters=request.hyperparameters
    )
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "Algorithm", "Accuracy (%)", "Precision (%)", "Recall (%)", 
        "F1-Score (%)", "Cohen's Kappa", "ROC-AUC", "MAE", "RMSE", "Execution Time (ms)", "Memory Used (MB)"
    ])
    
    for alg, metrics in results.items():
        writer.writerow([
            alg,
            round(metrics.accuracy * 100, 2),
            round(metrics.precision * 100, 2),
            round(metrics.recall * 100, 2),
            round(metrics.f1_score * 100, 2),
            round(metrics.cohen_kappa, 4),
            round(metrics.roc_auc, 4) if metrics.roc_auc is not None else "N/A",
            round(metrics.mae, 4),
            round(metrics.rmse, 4),
            round(metrics.execution_time_ms, 2),
            round(metrics.memory_used_mb, 4)
        ])
        
    output.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="comparison_benchmark_{request.dataset_id[:8]}.csv"'
    }
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers
    )
