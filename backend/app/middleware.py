import time
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Lightweight, memory-safe sliding window rate-limiting middleware.
    Limits clients to 100 requests per minute per IP.
    """
    def __init__(self, app, limit: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = {} # IP -> list of timestamps
        
    async def dispatch(self, request: Request, call_next):
        # Ignore docs and static assets from rate limiting to prevent UI lagging
        if request.url.path.startswith(("/docs", "/redoc", "/plots", "/openapi.json")):
            return await call_next(request)
            
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Filter old timestamps outside the sliding window
        timestamps = [t for t in self.requests.get(ip, []) if now - t < self.window_seconds]
        
        if len(timestamps) >= self.limit:
            logger.warning(f"Rate limit triggered for client IP: {ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down and try again later."
            )
            
        timestamps.append(now)
        self.requests[ip] = timestamps
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    HTTP security header injection middleware protecting against clickjacking,
    XSS injections, MIME sniffing, and forcing HTTPS connections.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
