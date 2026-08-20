"""소량의 개인별 작업 기록으로 다음 작업 후 Borg CR10을 예측한다.

실제 응답과 예측값을 분리해 저장하며, 최소 데이터와 연속 검증 오차 기준을
통과하기 전에는 설문을 계속 요청한다. 외부 ML 프레임워크 없이 NumPy ridge
회귀를 사용해 배포 환경과 테스트 환경에서 동일하게 동작한다.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np


MODEL_VERSION = "personal-ridge-v1"
MIN_TRAINING_RESPONSES = 3
READY_RESPONSE_COUNT = 8
READY_VALIDATION_COUNT = 5
READY_MAE_THRESHOLD = 1.0
RIDGE_ALPHA = 2.0

FEATURE_KEYS = (
    "work_minutes",
    "total_minutes",
    "driving_minutes",
    "unknown_minutes",
    "item_count",
    "labor_load",
    "team_size",
)


def _vector(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(key, 0.0) or 0.0) for key in FEATURE_KEYS], dtype=float)


def fit_ridge_parameters(
    history: list[tuple[Mapping[str, float], float]],
) -> Optional[dict]:
    """학습 결과를 JSON 컬럼에 저장할 수 있는 형태로 반환한다."""
    if len(history) < MIN_TRAINING_RESPONSES:
        return None

    x = np.vstack([_vector(features) for features, _ in history])
    y = np.asarray([float(borg) for _, borg in history], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_scaled = (x - mean) / scale
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y

    return {
        "feature_keys": list(FEATURE_KEYS),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": RIDGE_ALPHA,
    }


def predict_with_parameters(
    parameters: Optional[Mapping],
    current_features: Mapping[str, float],
) -> Optional[float]:
    """저장된 공통·개인 모델 파라미터로 Borg를 예측한다."""
    if not parameters:
        return None
    try:
        feature_keys = tuple(parameters.get("feature_keys") or FEATURE_KEYS)
        mean = np.asarray(parameters["mean"], dtype=float)
        scale = np.asarray(parameters["scale"], dtype=float)
        coefficients = np.asarray(parameters["coefficients"], dtype=float)
        if len(feature_keys) != len(mean) or len(mean) != len(scale):
            return None
        vector = np.asarray(
            [float(current_features.get(key, 0.0) or 0.0) for key in feature_keys],
            dtype=float,
        )
        current = (vector - mean) / scale
        prediction = float(np.concatenate([[1.0], current]) @ coefficients)
    except (KeyError, TypeError, ValueError):
        return None
    return round(min(10.0, max(0.0, prediction)), 1)


def _fit_predict(
    history: list[tuple[Mapping[str, float], float]],
    current_features: Mapping[str, float],
) -> Optional[float]:
    parameters = fit_ridge_parameters(history)
    return predict_with_parameters(parameters, current_features)


def predict_personal_fatigue(
    history: Iterable[dict],
    current_features: Mapping[str, float],
    fallback_parameters: Optional[Mapping] = None,
    fallback_model_version: Optional[str] = None,
) -> dict:
    """실제 응답을 우선 학습하고 부족할 때 저장된 초기 모델을 사용한다.

    설문 생략 여부는 테스트·사전 데이터가 아니라 해당 작업자의 실제 응답과
    사전 검증 오차만으로 결정한다.
    """
    ordered = list(history)
    training = [
        (row["features"], float(row["borg_cr10"]))
        for row in ordered
        if row.get("borg_cr10") is not None and row.get("features")
    ]
    validation_errors = [
        abs(float(row["borg_cr10"]) - float(row["predicted_borg_cr10"]))
        for row in ordered
        if row.get("borg_cr10") is not None and row.get("predicted_borg_cr10") is not None
    ]
    recent_errors = validation_errors[-READY_VALIDATION_COUNT:]
    validation_mae = round(float(np.mean(recent_errors)), 2) if recent_errors else None
    model_ready = (
        len(training) >= READY_RESPONSE_COUNT
        and len(recent_errors) >= READY_VALIDATION_COUNT
        and validation_mae is not None
        and validation_mae <= READY_MAE_THRESHOLD
    )
    personal_parameters = fit_ridge_parameters(training)
    prediction = predict_with_parameters(personal_parameters, current_features)
    prediction_source = "personal_actual" if prediction is not None else None
    model_version = MODEL_VERSION
    if prediction is None:
        prediction = predict_with_parameters(fallback_parameters, current_features)
        if prediction is not None:
            prediction_source = "stored_initial"
            model_version = fallback_model_version or MODEL_VERSION
    confidence = (
        "high" if model_ready
        else "medium" if len(training) >= 5 and prediction is not None
        else "low" if prediction is not None
        else "insufficient"
    )
    return {
        "predicted_borg_cr10": prediction,
        "prediction_confidence": confidence,
        "prediction_source": prediction_source or "insufficient",
        "model_version": model_version,
        "actual_response_count": len(training),
        "validation_count": len(validation_errors),
        "validation_mae": validation_mae,
        "model_ready": model_ready,
        "survey_required": not model_ready,
        "readiness_rule": {
            "minimum_actual_responses": READY_RESPONSE_COUNT,
            "minimum_validations": READY_VALIDATION_COUNT,
            "maximum_recent_mae": READY_MAE_THRESHOLD,
        },
    }
