import json
from urllib.request import Request, urlopen
from fastapi.testclient import TestClient
from api.main import app

PAYLOAD = {
    "id": "PATIENT_VERIFY_01",
    "age": 55,
    "sex": "Male",
    "cp": "atypical angina",
    "trestbps": 130.0,
    "chol": 240.0,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150.0,
    "exang": False,
    "oldpeak": 1.2,
    "slope": "flat",
    "ca": 0.0,
    "thal": "normal",
}


def call_container():
    data = json.dumps(PAYLOAD).encode("utf-8")
    request = Request(
        "http://localhost:8001/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("HEART DISEASE: LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local_response = client.post("/predict", json=PAYLOAD)

    container_response = call_container()

    local_data = local_response.json()
    local_pred = local_data["prediction"]
    container_pred = container_response["prediction"]

    local_prob = local_data["heart_disease_probability"]
    container_prob = container_response["heart_disease_probability"]

    print("\nLOCAL RESPONSE:")
    print(json.dumps(local_data, indent=2))

    print("\nCONTAINER RESPONSE:")
    print(json.dumps(container_response, indent=2))

    print(f"\nLocal prediction      : {local_pred}")
    print(f"Container prediction : {container_pred}")
    print(f"Local probability     : {local_prob}")
    print(f"Container probability: {container_prob}")

    match = (local_pred == container_pred) and (abs(local_prob - container_prob) < 1e-4)
    print(f"\nConsistency match     : {match}")

    if match:
        print("\n[SUCCESS] Heart Disease local and container predictions are 100% consistent.")
    else:
        print("\n[FAILURE] Predictions differ.")
        raise AssertionError("Container parity failed.")


if __name__ == "__main__":
    main()
