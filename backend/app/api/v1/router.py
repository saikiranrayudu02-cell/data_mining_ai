from fastapi import APIRouter
from app.api.v1.endpoints import dataset, classify, compare, export

api_router = APIRouter()

api_router.include_router(dataset.router, tags=["dataset"])
api_router.include_router(classify.router, prefix="/classify", tags=["classify"])
api_router.include_router(compare.router, prefix="/compare", tags=["compare"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
