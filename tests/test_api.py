import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


SAMPLE_HEALTHY_PATIENT = {
    "id": "P_001",
    "age": 45,
    "sex": "Female",
    "cp": "atypical angina",
    "trestbps": 120.0,
    "chol": 180.0,
    "fbs": False,
    "restecg": "normal",
    "thalch": 165.0,
    "exang": False,
    "oldpeak": 0.0,
    "slope": "upsloping",
    "ca": 0.0,
    "thal": "normal",
}

SAMPLE_AT_RISK_PATIENT = {
    "id": "P_002",
    "age": 67,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 160.0,
    "chol": 286.0,
    "fbs": True,
    "restecg": "lv hypertrophy",
    "thalch": 108.0,
    "exang": True,
    "oldpeak": 2.5,
    "slope": "flat",
    "ca": 3.0,
    "thal": "reversable defect",
}


def test_root_endpoint(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [307, 302, 200]
    if response.status_code in [307, 302]:
        assert response.headers["location"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "Heart Disease Risk Prediction API" in data["service"]
    assert data["model_name"] == "Heart_Disease_Production_Model"


def test_valid_single_prediction(client):
    response = client.post("/predict", json=SAMPLE_HEALTHY_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "heart_disease_probability" in data
    assert "risk_level" in data
    assert 0.0 <= data["heart_disease_probability"] <= 1.0


def test_at_risk_prediction(client):
    response = client.post("/predict", json=SAMPLE_AT_RISK_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["heart_disease_probability"] > 0.5


def test_invalid_categorical_field(client):
    invalid_patient = SAMPLE_HEALTHY_PATIENT.copy()
    invalid_patient["sex"] = "UnknownGender"
    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422


def test_invalid_blood_pressure_range(client):
    invalid_patient = SAMPLE_HEALTHY_PATIENT.copy()
    invalid_patient["trestbps"] = 999.0
    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422


def test_batch_prediction(client):
    batch_payload = {
        "patients": [SAMPLE_HEALTHY_PATIENT, SAMPLE_AT_RISK_PATIENT]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_patients"] == 2
    assert len(data["predictions"]) == 2
