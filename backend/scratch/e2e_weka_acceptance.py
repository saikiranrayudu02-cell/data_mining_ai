import urllib.request
import urllib.parse
import json
import uuid
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api/v1"
IRIS_PATH = Path("/Users/maggi/data_mine_pro/datasets/iris.arff")

def post_json(url, data_dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(data_dict).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def get_json(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def upload_file(url, file_path):
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    with open(file_path, "rb") as f:
        content = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def run_e2e_acceptance():
    print("=" * 60)
    print("WEKA EXPLORER END-TO-END ACCEPTANCE TEST")
    print("=" * 60)

    # 1. Upload iris.arff
    print("\n1. Testing ARFF Upload (/upload)...")
    status, upload_data = upload_file(f"{BASE_URL}/upload", IRIS_PATH)
    assert status in (200, 201), f"Upload failed: {upload_data}"
    dataset_id = upload_data["dataset_id"]
    print(f"✓ Upload successful! dataset_id: {dataset_id}")
    print(f"  Relation: {upload_data['relation_name']}, Instances: {upload_data['num_instances']}")

    # 2. Get Dataset Info
    print("\n2. Testing Dataset Info (/dataset/info)...")
    status, info = get_json(f"{BASE_URL}/dataset/info", {"dataset_id": dataset_id})
    assert status == 200, f"Info failed: {info}"
    print(f"✓ Dataset Relation: {info['relation_name']}")
    print(f"  Class Attribute: {info['class_attribute']}")
    print(f"  Class Distribution: {info['class_distribution']}")
    print(f"  Dataset MD5 Hash: {info['dataset_hash']}")
    assert info["num_instances"] == 150
    assert len(info["attributes"]) == 5

    # 3. Test Classifiers (J48, ID3, NaiveBayes, KNN)
    algorithms = ["J48", "ID3", "NaiveBayes", "KNN"]
    for alg in algorithms:
        print(f"\n3. Testing Classifier ({alg}) with 10-fold Cross-Validation...")
        params = {}
        if alg == "J48":
            params = {"confidence_threshold": 0.25, "min_instances": 2}
        elif alg == "ID3":
            params = {"min_instances": 1}
        elif alg == "KNN":
            params = {"n_neighbors": 1}

        status, res = post_json(f"{BASE_URL}/classify/train", {
            "dataset_id": dataset_id,
            "algorithm": alg,
            "target_attribute": "class",
            "evaluation_mode": "cross_validation",
            "folds": 10,
            "random_seed": 1,
            "hyperparameters": params
        })
        assert status == 200, f"Train {alg} failed: {res}"
        metrics = res["metrics"]

        print(f"✓ {alg} Classification Completed!")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}% | Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | RAE: {metrics['rae']:.2f}% | RRSE: {metrics['rrse']:.2f}%")
        print(f"  Correctly Classified: {metrics['correctly_classified_instances']} ({metrics['correctly_classified_pct']:.2f}%)")
        print(f"  Detailed Accuracy Classes Count: {len(metrics['detailed_accuracy_by_class'])}")

        if alg in ["J48", "ID3"]:
            print(f"  Rules Extracted: {len(res['rules'])}")
            print(f"  Tree Depth: {res['tree_depth']}, Leaves: {res['tree_leaf_nodes']}, Split Nodes: {res['tree_internal_nodes']}")
            assert res["tree_depth"] is not None
            assert len(metrics["entropy_stats"]) > 0

        assert len(metrics["raw_weka_output"]) > 100
        assert "=== Summary ===" in metrics["raw_weka_output"]
        assert "=== Confusion Matrix ===" in metrics["raw_weka_output"]

    # 4. Compare All Classifiers
    print("\n4. Testing Multi-Classifier Comparison Benchmark (/compare)...")
    status, comp = post_json(f"{BASE_URL}/compare/", {
        "dataset_id": dataset_id,
        "target_attribute": "class",
        "evaluation_mode": "cross_validation",
        "folds": 10,
        "random_seed": 1,
        "hyperparameters": {
            "J48": {"confidence_threshold": 0.25},
            "KNN": {"n_neighbors": 1}
        }
    })
    assert status == 200, f"Compare failed: {comp}"
    print(f"✓ Multi-Classifier Comparison Benchmark Completed!")
    print(f"  Best Algorithm: {comp['best_algorithm']}")
    print(f"  Worst Algorithm: {comp['worst_algorithm']}")
    print(f"  Algorithm Rankings: {comp['rankings']}")
    print(f"  Ranking Rationale: {comp['ranking_reason']}")
    assert comp["best_algorithm"] in algorithms
    assert comp["worst_algorithm"] in algorithms

    print("\n" + "=" * 60)
    print("ALL END-TO-END WEKA ACCEPTANCE STEPS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_acceptance()
