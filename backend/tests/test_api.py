from fastapi.testclient import TestClient
from app.main import app
import tempfile
import os

client = TestClient(app)

def test_read_root():
    """
    Test root health check endpoint returns 200 and correct status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_dataset_upload_wrong_format():
    """
    Ensure upload endpoint rejects non-arff files with bad request 400.
    """
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only .arff files are supported" in response.json()["detail"]

def test_arff_upload_info_and_preview_flow():
    """
    Verify complete flow:
    1. Upload valid ARFF file.
    2. Retrieve dataset details via /dataset/info.
    3. Retrieve dataset preview via /dataset/preview.
    """
    arff_content = """@relation test_rel
@attribute x numeric
@attribute class {yes, no}
@data
1.5,yes
2.5,no
?,yes
"""
    fd, temp_path = tempfile.mkstemp(suffix=".arff")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(arff_content)
            
        with open(temp_path, "rb") as f:
            response = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})
            
        assert response.status_code == 201
        res_json = response.json()
        assert "dataset_id" in res_json
        
        dataset_id = res_json["dataset_id"]
        
        # Test GET /dataset/info
        info_resp = client.get(f"/dataset/info?dataset_id={dataset_id}")
        assert info_resp.status_code == 200
        
        # Test GET /dataset/preview
        prev_resp = client.get(f"/dataset/preview?dataset_id={dataset_id}&limit=2")
        assert prev_resp.status_code == 200
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_train_classifiers_endpoint():
    """
    Test POST /train dynamic model training endpoint.
    """
    arff_content = """@relation test_train_rel
@attribute temperature numeric
@attribute play {yes, no}
@data
85,yes
80,no
83,yes
70,yes
68,no
"""
    fd, temp_path = tempfile.mkstemp(suffix=".arff")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(arff_content)
            
        with open(temp_path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})
            
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]
        
        # Train J48
        train_resp = client.post("/train", json={
            "dataset_id": dataset_id,
            "algorithm": "J48",
            "target_attribute": "play",
            "test_split": 0.2,
            "hyperparameters": {"min_instances": 1}
        })
        
        assert train_resp.status_code == 200
        train_json = train_resp.json()
        assert "model_id" in train_json
        assert train_json["algorithm"] == "J48"
        assert "metrics" in train_json
        assert "accuracy" in train_json["metrics"]
        assert "cohen_kappa" in train_json["metrics"]
        assert "classification_report" in train_json["metrics"]
        assert "confusion_matrix_plot_url" in train_json["metrics"]
        
        # Verify tree metrics
        assert "tree_depth" in train_json
        assert "tree_leaf_nodes" in train_json
        assert "tree_internal_nodes" in train_json
        assert train_json["tree_depth"] is not None
        assert train_json["tree_leaf_nodes"] >= 1
        
        # Verify static plot file endpoints return 200
        cm_url = train_json["metrics"]["confusion_matrix_plot_url"]
        cm_resp = client.get(cm_url)
        assert cm_resp.status_code == 200
        assert cm_resp.headers["content-type"] == "image/png"
        
        # Verify static tree PNG returns 200 if generated
        tree_url = train_json.get("tree_png_url")
        if tree_url:
            tree_resp = client.get(tree_url)
            assert tree_resp.status_code == 200
            assert tree_resp.headers["content-type"] == "image/png"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_compare_classifiers_endpoint():
    """
    Verify POST /compare runs all 4 models and returns rankings, best, and worst algorithm tags.
    """
    arff_content = """@relation test_compare_rel
@attribute temperature numeric
@attribute play {yes, no}
@data
85,yes
80,no
83,yes
70,yes
68,no
"""
    fd, temp_path = tempfile.mkstemp(suffix=".arff")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(arff_content)
            
        with open(temp_path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})
            
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]
        
        # Call Compare
        comp_resp = client.post("/compare", json={
            "dataset_id": dataset_id,
            "target_attribute": "play",
            "test_split": 0.2,
            "hyperparameters": {
                "KNN": {"n_neighbors": 3},
                "J48": {"confidence_threshold": 0.25}
            }
        })
        
        assert comp_resp.status_code == 200
        comp_json = comp_resp.json()
        assert comp_json["dataset_id"] == dataset_id
        assert "results" in comp_json
        assert "ID3" in comp_json["results"]
        assert "J48" in comp_json["results"]
        assert "best_algorithm" in comp_json
        assert "worst_algorithm" in comp_json
        assert "rankings" in comp_json
        assert len(comp_json["rankings"]) == 4
        
        # Verify memory delta exists
        assert "memory_used_mb" in comp_json["results"]["ID3"]
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_compare_export_csv_endpoint():
    """
    Verify POST /compare/export-csv returns streaming attachment.
    """
    arff_content = """@relation test_csv_rel
@attribute x numeric
@attribute class {yes, no}
@data
1.5,yes
2.5,no
3.5,yes
4.5,no
"""
    fd, temp_path = tempfile.mkstemp(suffix=".arff")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(arff_content)
            
        with open(temp_path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})
            
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]
        
        # Call Export CSV
        csv_resp = client.post("/compare/export-csv", json={
            "dataset_id": dataset_id,
            "target_attribute": "class",
            "test_split": 0.2
        })
        
        assert csv_resp.status_code == 200
        assert csv_resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in csv_resp.headers["content-disposition"]
        
        # Check CSV content structure
        csv_text = csv_resp.text
        assert "Algorithm" in csv_text
        assert "Memory Used" in csv_text
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_export_report_endpoints():
    """
    Verify POST /export generates PDF, HTML, and JSON reports on the fly
    and GET /export/download returns them successfully.
    """
    arff_content = """@relation test_export_rel
@attribute x numeric
@attribute class {yes, no}
@data
1.5,yes
2.5,no
3.5,yes
4.5,no
"""
    fd, temp_path = tempfile.mkstemp(suffix=".arff")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(arff_content)
            
        with open(temp_path, "rb") as f:
            upload_resp = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})
            
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]
        
        # Train J48 model first
        train_resp = client.post("/train", json={
            "dataset_id": dataset_id,
            "algorithm": "J48",
            "target_attribute": "class",
            "test_split": 0.2
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["model_id"]
        
        # 1. Test PDF Export
        export_resp = client.post("/export", json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "format": "pdf"
        })
        assert export_resp.status_code == 200
        pdf_json = export_resp.json()
        assert "download_url" in pdf_json
        
        # Download PDF file
        dl_resp = client.get(pdf_json["download_url"])
        assert dl_resp.status_code == 200
        
        # 2. Test HTML Export
        export_resp_html = client.post("/export", json={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "format": "html"
        })
        assert export_resp_html.status_code == 200
        html_json = export_resp_html.json()
        
        # Download HTML file
        dl_resp_html = client.get(html_json["download_url"])
        assert dl_resp_html.status_code == 200
        assert "<!DOCTYPE html>" in dl_resp_html.text
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
