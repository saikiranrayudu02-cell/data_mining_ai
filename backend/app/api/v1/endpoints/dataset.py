from fastapi import APIRouter, UploadFile, File, HTTPException, status, Query
from app.schemas.dataset import DatasetUploadResponse, DatasetInfoResponse, DatasetPreviewResponse
from app.ml.arff_parser import ARFFParser
from app.config import settings
import uuid
import os
from pathlib import Path

router = APIRouter()

@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and validate a .arff file.
    Saves the file to disk and runs parser checks.
    """
    if not file.filename.endswith(".arff"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .arff files are supported."
        )
        
    dataset_id = str(uuid.uuid4())
    saved_filename = f"{dataset_id}.arff"
    file_path = settings.UPLOAD_DIR / saved_filename
    
    try:
        # Save file to uploads directory
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # Parse data using scientific scipy engine to validate structural integrity
        parsed = ARFFParser.parse(file_path)
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            relation_name=parsed["relation"],
            num_attributes=len(parsed["attributes"]),
            num_instances=parsed["num_instances"],
            message="Dataset uploaded, parsed, and validated successfully"
        )
    except ValueError as ve:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ARFF structure validation error: {str(ve)}"
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal system failure processing dataset: {str(e)}"
        )

@router.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_info(dataset_id: str = Query(..., description="Unique dataset identifier")):
    """
    Load dataset schema information, attribute types, row count, target class,
    and missing values count per column.
    """
    file_path = settings.UPLOAD_DIR / f"{dataset_id}.arff"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found. Please upload the dataset first."
        )
        
    try:
        parsed = ARFFParser.parse(file_path)
        return DatasetInfoResponse(
            dataset_id=dataset_id,
            relation_name=parsed["relation"],
            num_attributes=len(parsed["attributes"]),
            num_instances=parsed["num_instances"],
            attributes=[
                {
                    "name": attr["name"],
                    "type": attr["type"],
                    "values": attr["values"]
                }
                for attr in parsed["attributes"]
            ],
            class_attribute=parsed["class_attribute"],
            class_distribution=parsed.get("class_distribution", {}),
            missing_values_count=parsed["missing_values"],
            dataset_hash=parsed.get("dataset_hash", "")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset details: {str(e)}"
        )

@router.get("/dataset/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str = Query(..., description="Unique dataset identifier"),
    limit: int = Query(10, ge=1, le=100, description="Number of instances to preview")
):
    """
    Returns lists of raw records alongside preprocessed records (imputed, encoded, normalized)
    for comparison and preview in the UI tables.
    """
    file_path = settings.UPLOAD_DIR / f"{dataset_id}.arff"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found. Please upload the dataset first."
        )
        
    try:
        parsed = ARFFParser.parse(file_path)
        df_raw = parsed["dataframe"]
        attributes = parsed["attributes"]
        
        # Take preview slice of raw DataFrame
        df_raw_slice = df_raw.head(limit)
        
        # Build Preprocessing pipeline
        # 1. Impute missing values
        df_imputed = ARFFParser.impute_missing_values(df_raw, attributes)
        # 2. Encode categorical types
        df_encoded, _ = ARFFParser.encode_categorical(df_imputed, attributes)
        # 3. Normalize numeric types
        df_processed = ARFFParser.normalize_numerical(df_encoded, attributes)
        
        # Take preview slice of preprocessed DataFrame
        df_processed_slice = df_processed.head(limit)
        
        # Format lists of dictionaries replacing NaN with None for json compatibility
        raw_list = df_raw_slice.replace({float('nan'): None}).to_dict(orient="records")
        processed_list = df_processed_slice.to_dict(orient="records")
        
        return DatasetPreviewResponse(
            dataset_id=dataset_id,
            columns=list(df_raw.columns),
            raw_data=raw_list,
            processed_data=processed_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dataset preview: {str(e)}"
        )
