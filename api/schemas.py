from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field


class HeartDiseasePatientInput(BaseModel):
    id: Optional[str] = "PATIENT_001"
    age: int = Field(ge=18, le=120, description="Age in years")
    sex: Literal["Male", "Female"] = Field(description="Patient gender")
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = Field(
        description="Chest pain type"
    )
    trestbps: float = Field(ge=50.0, le=250.0, description="Resting blood pressure in mm Hg")
    chol: float = Field(ge=50.0, le=700.0, description="Serum cholesterol in mg/dl")
    fbs: Union[bool, str, int] = Field(default=False, description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal["normal", "lv hypertrophy", "st-t abnormality"] = Field(
        default="normal", description="Resting electrocardiographic results"
    )
    thalch: float = Field(ge=50.0, le=250.0, description="Maximum heart rate achieved")
    exang: Union[bool, str, int] = Field(default=False, description="Exercise induced angina")
    oldpeak: float = Field(ge=-5.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: Literal["upsloping", "flat", "downsloping"] = Field(
        default="flat", description="Slope of the peak exercise ST segment"
    )
    ca: float = Field(ge=0.0, le=4.0, default=0.0, description="Number of major vessels (0-3) colored by flourosopy")
    thal: Literal["normal", "fixed defect", "reversable defect"] = Field(
        default="normal", description="Thalassemia stress test defect status"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(description="Binary prediction: 0 for No Heart Disease, 1 for Presence of Heart Disease")
    heart_disease_probability: float = Field(description="Predicted risk probability (0.0 to 1.0)")
    diagnosis: str = Field(description="Clinical diagnosis summary")
    risk_level: str = Field(description="Assessed risk level: Low, Moderate, or High")


class BatchPredictionRequest(BaseModel):
    patients: List[HeartDiseasePatientInput] = Field(min_length=1, description="List of patient diagnostic records")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_patients: int
