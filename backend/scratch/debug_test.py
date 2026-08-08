import os
import tempfile
from fastapi.testclient import TestClient
from app.main import app
import traceback

client = TestClient(app, raise_server_exceptions=False)

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
with os.fdopen(fd, "w") as f:
    f.write(arff_content)

with open(temp_path, "rb") as f:
    upload_resp = client.post("/upload", files={"file": ("test.arff", f, "application/octet-stream")})

print("Upload status:", upload_resp.status_code)
dataset_id = upload_resp.json()["dataset_id"]

train_resp = client.post("/train", json={
    "dataset_id": dataset_id,
    "algorithm": "J48",
    "target_attribute": "play",
    "test_split": 0.2,
    "hyperparameters": {"min_instances": 1}
})

print("Train status:", train_resp.status_code)
print("Train json:", train_resp.json())
