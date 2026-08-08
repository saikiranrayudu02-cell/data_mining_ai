from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from app.schemas.dataset import ExportRequest, ExportResponse
from app.ml.arff_parser import ARFFParser
from app.ml.report_generator import ReportGenerator
from app.ml.rule_extractor import RuleExtractor
from app.config import settings
import joblib
import json
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=ExportResponse, status_code=status.HTTP_200_OK)
async def export_report(request: ExportRequest):
    """
    Generate and export a classification report.
    Supports PDF, HTML, and JSON formats on the fly.
    """
    if not request.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required"
        )
        
    if not request.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model ID is required to export single model report"
        )
        
    model_path = settings.MODEL_DIR / f"{request.model_id}.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trained model artifact not found. Please train the model first."
        )
        
    arff_path = settings.UPLOAD_DIR / f"{request.dataset_id}.arff"
    if not os.path.exists(arff_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found."
        )
        
    try:
        # 1. Load serialized model wrapper and parsed arff metadata
        data = joblib.load(model_path)
        clf = data["model"]
        metrics = data["metrics"]
        
        parsed = ARFFParser.parse(arff_path)
        target_col = parsed["class_attribute"] or "class"
        feature_names = [attr["name"] for attr in parsed["attributes"] if attr["name"] != target_col]
        
        # 2. Extract rules
        rules = RuleExtractor.extract_rules(clf, feature_names, target_col)
        
        # 3. Setup visual plots image paths
        cm_path = str(settings.PLOT_DIR / f"{request.model_id}_cm.png")
        roc_path = str(settings.PLOT_DIR / f"{request.model_id}_roc.png")
        tree_path = str(settings.PLOT_DIR / f"{request.model_id}_tree.png")
        
        # Verify if images exist on disk (ROC is optional for multiclass)
        cm_img = cm_path if os.path.exists(cm_path) else None
        roc_img = roc_path if os.path.exists(roc_path) else None
        tree_img = tree_path if os.path.exists(tree_path) else None
        
        dataset_info = {
            "relation_name": parsed["relation"],
            "num_instances": parsed["num_instances"],
            "num_attributes": len(parsed["attributes"]),
            "class_attribute": target_col
        }
        
        # 4. Generate the export file cased
        filename = f"{request.model_id}_report.{request.format}"
        output_path = settings.EXPORT_DIR / filename
        
        algorithm_name = getattr(clf, "__class__").__name__.replace("Classifier", "")
        
        if request.format == "pdf":
            ReportGenerator.generate_pdf_report(
                dataset_info=dataset_info,
                algorithm_name=algorithm_name,
                metrics=metrics if isinstance(metrics, dict) else metrics.model_dump(),
                rules=rules,
                cm_image_path=cm_img,
                roc_image_path=roc_img,
                tree_image_path=tree_img,
                output_path=str(output_path)
            )
        elif request.format == "html":
            # Map URLs relative to browser serving path
            cm_url = f"/plots/{request.model_id}_cm.png" if cm_img else None
            roc_url = f"/plots/{request.model_id}_roc.png" if roc_img else None
            tree_url = f"/plots/{request.model_id}_tree.png" if tree_img else None
            
            ReportGenerator.generate_html_report(
                dataset_info=dataset_info,
                algorithm_name=algorithm_name,
                metrics=metrics if isinstance(metrics, dict) else metrics.model_dump(),
                rules=rules,
                cm_image_path=cm_img,
                roc_image_path=roc_img,
                tree_image_path=tree_img,
                output_path=str(output_path)
            )
        elif request.format == "json":
            report_data = {
                "dataset": dataset_info,
                "algorithm": algorithm_name,
                "metrics": metrics if isinstance(metrics, dict) else metrics.model_dump(),
                "rules": rules
            }
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported export format: {request.format}"
            )
            
        download_url = f"/api/v1/export/download/{filename}"
        return ExportResponse(
            download_url=download_url,
            format=request.format,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Failed to generate report export: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report export: {str(e)}"
        )

@router.get("/download/{filename}")
async def download_exported_file(filename: str):
    """
    Stream the generated PDF/HTML/JSON report to the client as an attachment.
    """
    file_path = settings.EXPORT_DIR / filename
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exported report file not found"
        )
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )
