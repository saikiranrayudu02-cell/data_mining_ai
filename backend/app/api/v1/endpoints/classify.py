from fastapi import APIRouter, HTTPException, status
from pathlib import Path
from app.schemas.dataset import ClassifyRequest, ClassifyResponse
from app.ml.arff_parser import ARFFParser
from app.ml.trainer.model_trainer import ModelTrainer
from app.ml.tree_visualizer import TreeVisualizer
from app.ml.rule_extractor import RuleExtractor
from app.config import settings
import os
import glob

router = APIRouter()

@router.post("/train", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
async def train_model(request: ClassifyRequest):
    """
    Train a specified classifier dynamically (J48, ID3, NaiveBayes, KNN) on the uploaded dataset.
    Saves the model using joblib and returns complete validation metrics.
    """
    file_path = None
    
    # 1. Resolve dataset
    if request.dataset_id:
        file_path = settings.UPLOAD_DIR / f"{request.dataset_id}.arff"
    else:
        # Fallback: Get most recently uploaded dataset
        arff_files = glob.glob(os.path.join(settings.UPLOAD_DIR, "*.arff"))
        if not arff_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No dataset available. Please upload a dataset first."
            )
        # Sort by creation time
        arff_files.sort(key=os.path.getmtime)
        file_path = Path(arff_files[-1])

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Specified dataset file was not found on disk."
        )

    try:
        # 2. Parse ARFF file structure
        parsed = ARFFParser.parse(file_path)
        df_raw = parsed["dataframe"]
        attributes = parsed["attributes"]
        
        # 3. Handle missing values first using imputer
        df_clean = ARFFParser.impute_missing_values(df_raw, attributes)
        
        # Determine target attribute
        target_col = request.target_attribute or parsed["class_attribute"]
        if not target_col or target_col not in df_clean.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target class attribute '{target_col}' not found in dataset."
            )
            
        # 4. Train and serialize model dynamically
        clf, metrics, model_id = ModelTrainer.train_and_evaluate(
            df=df_clean,
            target_col=target_col,
            algorithm_name=request.algorithm,
            evaluation_mode=request.evaluation_mode,
            percentage_split=request.percentage_split,
            folds=request.folds,
            random_seed=request.random_seed,
            test_split=request.test_split,
            hyperparameters=request.hyperparameters,
            relation_name=parsed.get("relation", "dataset")
        )
        
        # 5. Extract rules & tree structure if decision tree algorithm
        rules = []
        tree_structure = None
        tree_png_url = None
        tree_depth = None
        tree_leaf_nodes = None
        tree_internal_nodes = None
        
        if request.algorithm in ["ID3", "J48"]:
            # Feature list excluding target
            feature_names = [attr["name"] for attr in attributes if attr["name"] != target_col]
            rules = RuleExtractor.extract_rules(clf, feature_names, target_col)
            tree_structure = TreeVisualizer.to_react_flow(clf, feature_names)
            
            # Calculate Tree Metrics
            tree_depth = TreeVisualizer.get_tree_depth(clf.root)
            tree_leaf_nodes = TreeVisualizer.get_leaf_count(clf.root)
            tree_internal_nodes = TreeVisualizer.get_split_count(clf.root)
            
            # Export Graphviz DOT & Compile PNG
            dot_str = TreeVisualizer.to_graphviz(clf, feature_names)
            tree_filename = f"{model_id}_tree.png"
            tree_path = str(settings.PLOT_DIR / tree_filename)
            TreeVisualizer.compile_dot_to_png(dot_str, tree_path)
            
            # Check if compiled output was written to disk
            if os.path.exists(tree_path):
                tree_png_url = f"/plots/{tree_filename}"

        return ClassifyResponse(
            model_id=model_id,
            algorithm=request.algorithm,
            metrics=metrics,
            rules=rules,
            tree=tree_structure,
            tree_png_url=tree_png_url,
            tree_depth=tree_depth,
            tree_leaf_nodes=tree_leaf_nodes,
            tree_internal_nodes=tree_internal_nodes
        )
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model training input failure: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete model training: {str(e)}"
        )
