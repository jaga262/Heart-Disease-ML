import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import (
    HeartDiseasePatientInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from api.monitoring_logger import log_inference

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load verified production model and preprocessor
    model_loader.load()
    yield
    # Shutdown
    print("[INFO] Heart Disease Risk Prediction API shutting down...")


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production ML Microservice for Heart Disease Risk Evaluation & Clinical Decision Support",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(location) for location in error["loc"] if location != "body"]
        field = ".".join(locations)
        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name or "Heart_Disease_Production_Model",
        model_version=model_loader.model_version or 1,
        run_id=model_loader.run_id or "",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Patient input validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail),
        },
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


@app.get("/metadata")
def metadata():
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
        "metadata": model_loader.metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartDiseasePatientInput):
    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()
    data_dict = data.model_dump()

    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        df = pd.DataFrame([data_dict])
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)

        prediction = int(model_loader.model.predict(X)[0])
        probability = float(model_loader.model.predict_proba(X)[0][1])

        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        diagnosis = (
            "Presence of Heart Disease Indicated"
            if prediction == 1
            else "No Heart Disease Detected"
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="success",
            prediction=prediction,
            probability=probability,
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        return PredictionResponse(
            prediction=prediction,
            heart_disease_probability=round(probability, 4),
            diagnosis=diagnosis,
            risk_level=risk_level,
        )

    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {exc}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request_payload: BatchPredictionRequest):
    patients_data = [p.model_dump() for p in request_payload.patients]
    if not patients_data:
        raise HTTPException(status_code=400, detail="Patients list cannot be empty")

    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        df = pd.DataFrame(patients_data)
        df = clean_prediction_data(df)
        X = model_loader.preprocessor.transform(df)

        preds = model_loader.model.predict(X)
        probs = model_loader.model.predict_proba(X)[:, 1]

        responses = []
        for p, prob in zip(preds, probs):
            pred_int = int(p)
            prob_flt = float(prob)
            risk = "Low" if prob_flt < 0.35 else ("Moderate" if prob_flt < 0.65 else "High")
            diag = "Presence of Heart Disease Indicated" if pred_int == 1 else "No Heart Disease Detected"

            responses.append(
                PredictionResponse(
                    prediction=pred_int,
                    heart_disease_probability=round(prob_flt, 4),
                    diagnosis=diag,
                    risk_level=risk,
                )
            )

        return BatchPredictionResponse(
            predictions=responses,
            total_patients=len(responses),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")
