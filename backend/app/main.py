from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.v1.router import api_router
from app.api.v1.endpoints import dataset, classify, compare, export
from app.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend ML classification service for ID3, J48, Naive Bayes, and KNN algorithms on ARFF datasets.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register security and rate limiter middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

# Register API routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(dataset.router)
app.include_router(classify.router)
app.include_router(compare.router, prefix="/compare")
app.include_router(export.router, prefix="/export")

# Serve generated plots statically
app.mount("/plots", StaticFiles(directory=str(settings.PLOT_DIR)), name="plots")

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "docs": "/docs"
    }
