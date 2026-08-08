# DataMine AI Classifier

A production-ready full-stack machine learning workbench for parsing ARFF (Attribute-Relation File Format) datasets, running classification algorithms (ID3, J48, Naive Bayes, KNN), and generating visual analytics performance reports.

---

## Technical Stack Overview

- **Frontend**: Next.js 15 (App Router), TypeScript, Vanilla CSS Modules (No Tailwind), React Flow, Chart.js.
- **Backend**: Python FastAPI, scikit-learn, Scipy, ReportLab (PDF compiler), Uvicorn.
- **Middlewares**: Custom ASGI IP rate-limiter, Security Headers injector.
- **Testing**: Jest + React Testing Library (Frontend), Pytest + Starlette TestClient (Backend).
- **DevOps**: Docker, Docker Compose orchestration, GitHub Actions CI pipelines.

---

## Folder Structure

```text
data_mine_pro/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/  # API router handlers (dataset, classify, compare, export)
│   │   ├── ml/                 # Machine learning models (ID3, J48, Evaluator, ReportGenerator)
│   │   ├── schemas/            # Pydantic models
│   │   ├── storage/            # Models, uploads, plots, exports directories
│   │   ├── main.py             # FastAPI startup & middleware configuration
│   │   └── middleware.py       # Rate limiter and security headers
│   ├── tests/                  # Pytest verification suites
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                # Next.js App routing views (upload, preview, classify, compare)
│   │   ├── components/         # Premium layout widgets (Navbar, Sidebar)
│   │   └── services/           # Api service fetch clients
│   ├── tests/                  # Jest unit tests
│   ├── Dockerfile
│   └── package.json
└── docker-compose.yml          # Container orchestration linkage
```

---

## Quick Start via Docker Compose (Recommended)

To build and boot both backend and frontend services inside isolated Docker containers, simply run:

```bash
docker-compose up --build
```

- **Frontend Portal**: `http://localhost:3000`
- **FastAPI Backend Server**: `http://localhost:8000`
- **API Documentation (Swagger)**: `http://localhost:8000/docs`

---

## Local Development Installation

### 1. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn app.main:app --reload
   ```

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Run the development proxy server:
   ```bash
   npm run dev
   ```

---

## API Endpoints List

- `POST /upload`: Upload `.arff` datasets, validate structures, and save files.
- `GET /dataset/info/{dataset_id}`: Fetch relation name, attributes list, row instances, and labels.
- `GET /dataset/preview/{dataset_id}`: Fetch paginated previews of raw and preprocessed data arrays.
- `POST /train`: Fit a classifier (J48, ID3, Naive Bayes, KNN) on a dataset, generate confusion matrix and ROC curve plots, extract IF-THEN rules, and serialize the model to disk.
- `POST /compare`: Run cross-validation benchmarks comparing accuracy, duration, and memory footprint across all classifiers side-by-side.
- `POST /compare/export-csv`: Export benchmarking reports as a spreadsheet file attachment.
- `POST /export`: Generate printable HTML, PDF, or JSON reports including metadata and plots.
- `GET /export/download/{filename}`: Download generated report attachments.

---

## Production Configurations

### Rate Limiting
- Custom `RateLimitMiddleware` automatically blocks requests exceeding **120 requests/minute per IP** (exempting static files and Swagger docs paths) returning `429 Too Many Requests`.

### Security Headers
- `SecurityHeadersMiddleware` injects strict security policy headers protecting clients against MIME sniffing, clickjacking, and XSS:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains`

### Caching
- Added `functools.lru_cache` to `ARFFParser.parse` to avoid parsing raw metadata files repeatedly, optimizing benchmarking speeds.

---

## Testing & Checks

Run backend unit tests:
```bash
cd backend
venv/bin/pytest
```

Run frontend unit tests:
```bash
cd frontend
npm run test
```

Typecheck frontend:
```bash
npx tsc --noEmit
```
