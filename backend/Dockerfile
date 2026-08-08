FROM python:3.11-slim

# Install system dependencies, including graphviz
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . /workspace/

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn production runner
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
