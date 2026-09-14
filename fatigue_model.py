"""소량의 개인별 작업 기록으로 다음 작업 후 Borg CR10을 예측한다.

실제 응답과 예측값을 분리해 저장하며, 최소 데이터와 연속 검증 오차 기준을
통과하기 전에는 설문을 계속 요청한다. 외부 ML 프레임워크 없이 NumPy ridge
회귀를 사용해 배포 환경과 테스트 환경에서 동일하게 동작한다.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Mapping, Optional

import numpy as np


MODEL_VERSION = "personal-ridge-v1"
RECOVERY_MODEL_VERSION = "personal-recovery-ridge-v3"
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
CUMULATIVE_FATIGUE_FEATURE_KEY = "cumulative_fatigue"
RECOVERY_MINUTES_FEATURE_KEY = "recovery_minutes"
LEGACY_RECOVERY_FEATURE_KEY = "recovered_daily_load"
RECOVERY_FEATURE_KEYS = FEATURE_KEYS + (
    CUMULATIVE_FATIGUE_FEATURE_KEY,
    RECOVERY_MINUTES_FEATURE_KEY,
)
DEFAULT_RECOVERY_HALF_LIFE_MINUTES = 120.0
WORKER_RECOVERY_HALF_LIFE_MINUTES = {
    "김경언": 60.0,
    "최현": 90.0,
    "강경래": 120.0,
    "정하람": 180.0,
    "박우민": 360.0,
}
RECOVERY_HALF_LIFE_CANDIDATES = (60.0, 90.0, 120.0, 180.0, 240.0, 360.0)
MIN_RECOVERY_CALIBRATION_TRANSITIONS = 3


def recovery_half_life_for_worker(
    worker_name: Optional[str],
    fallback: float = DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
) -> float:
    """작업자별 운영 회복 반감기를 반환한다.

    반감기가 짧을수록 회복 속도가 빠르다. 지정된 5명 외 작업자는 전달된
    fallback 또는 기본 120분을 사용한다.
    """
    normalized_name = str(worker_name or "").strip()
    return WORKER_RECOVERY_HALF_LIFE_MINUTES.get(
        normalized_name,
        max(1.0, float(fallback)),
    )


def recovery_rate_per_minute(half_life_minutes: float) -> float:
    """지수 감쇠식 ``exp(-k*t)``의 분당 회복계수 k를 반환한다."""
    return math.log(2.0) / max(1.0, float(half_life_minutes))


def _vector(
    features: Mapping[str, float],
    feature_keys: Iterable[str] = FEATURE_KEYS,
) -> np.ndarray:
    return np.asarray(
        [float(features.get(key, 0.0) or 0.0) for key in feature_keys],
        dtype=float,
    )


def fit_ridge_parameters(
    history: list[tuple[Mapping[str, float], float]],
    feature_keys: Iterable[str] = FEATURE_KEYS,
) -> Optional[dict]:
    """학습 결과를 JSON 컬럼에 저장할 수 있는 형태로 반환한다."""
    if len(history) < MIN_TRAINING_RESPONSES:
        return None

    keys = tuple(feature_keys)
    x = np.vstack([_vector(features, keys) for features, _ in history])
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
        "feature_keys": list(keys),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": RIDGE_ALPHA,
    }


def _as_datetime(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except ValueError:
            return None
    return None


def fatigue_decay(
    value: float,
    elapsed_minutes: float,
    half_life_minutes: float = DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
) -> float:
    """반감기 기반의 단순 회복 곡선으로 남아 있는 부하를 계산한다.

    이 값은 생리학적 진단값이 아니라, 같은 날 작업 사이의 휴식 효과를 모델 입력에
    일관되게 반영하기 위한 운영용 상태값이다.
    """
    half_life = max(1.0, float(half_life_minutes))
    elapsed = max(0.0, float(elapsed_minutes))
    return max(0.0, float(value)) * (0.5 ** (elapsed / half_life))


def _effective_borg(record: Mapping) -> Optional[float]:
    value = record.get("borg_cr10")
    if value is None:
        value = record.get("predicted_borg_cr10")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def recovered_daily_load(
    history: Iterable[Mapping],
    current_started_at,
    half_life_minutes: float = DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
) -> float:
    """현재 출동 직전까지 남은 당일 누적 Borg 피로도를 반환한다.

    같은 날 완료된 이전 출동의 Borg를 각각 현재 출동 시작 시각까지 감쇠해
    합산한다. 첫 출동과 날짜가 바뀐 첫 출동은 0이며 현재 출동의 Borg는 쓰지 않는다.
    """
    started_at = _as_datetime(current_started_at)
    if started_at is None:
        return 0.0
    total = 0.0
    for record in history:
        completed_at = _as_datetime(record.get("completed_at"))
        if (
            completed_at is None
            or completed_at > started_at
            or completed_at.date() != started_at.date()
        ):
            continue
        borg = _effective_borg(record)
        if borg is None:
            continue
        elapsed = (started_at - completed_at).total_seconds() / 60.0
        total += fatigue_decay(borg, elapsed, half_life_minutes)
    return round(total, 4)


def recovery_minutes_since_previous(
    history: Iterable[Mapping],
    current_started_at,
) -> float:
    """당일 직전 출동 완료부터 현재 출동 시작까지의 시간을 분으로 반환한다."""
    started_at = _as_datetime(current_started_at)
    if started_at is None:
        return 0.0
    prior_completions = [
        completed_at
        for record in history
        if (completed_at := _as_datetime(record.get("completed_at"))) is not None
        and completed_at <= started_at
        and completed_at.date() == started_at.date()
    ]
    if not prior_completions:
        return 0.0
    elapsed = (started_at - max(prior_completions)).total_seconds() / 60.0
    return round(max(0.0, elapsed), 4)


def with_recovery_features(
    features: Mapping[str, float],
    history: Iterable[Mapping],
    current_started_at,
    half_life_minutes: float = DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
) -> dict:
    result = dict(features)
    started_at = _as_datetime(current_started_at)
    if started_at is None:
        cumulative_fatigue = float(
            result.get(
                CUMULATIVE_FATIGUE_FEATURE_KEY,
                result.get(LEGACY_RECOVERY_FEATURE_KEY, 0.0),
            )
            or 0.0
        )
        recovery_minutes = float(result.get(RECOVERY_MINUTES_FEATURE_KEY, 0.0) or 0.0)
    else:
        history_rows = list(history)
        cumulative_fatigue = recovered_daily_load(
            history_rows,
            started_at,
            half_life_minutes,
        )
        recovery_minutes = recovery_minutes_since_previous(history_rows, started_at)
    result[CUMULATIVE_FATIGUE_FEATURE_KEY] = cumulative_fatigue
    result[RECOVERY_MINUTES_FEATURE_KEY] = recovery_minutes
    # 기존 API/저장 모델 호환을 위해 같은 값을 별칭으로 유지한다.
    result[LEGACY_RECOVERY_FEATURE_KEY] = cumulative_fatigue
    return result


def with_recovery_feature(
    features: Mapping[str, float],
    history: Iterable[Mapping],
    current_started_at,
    half_life_minutes: float = DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
) -> dict:
    """이전 단수형 함수명의 하위 호환 별칭."""
    return with_recovery_features(
        features,
        history,
        current_started_at,
        half_life_minutes,
    )


def _recovery_training_rows(
    records: Iterable[Mapping],
    half_life_minutes: Optional[float],
) -> list[dict]:
    grouped: dict[str, list[tuple[int, Mapping]]] = {}
    for index, record in enumerate(records):
        worker_name = str(record.get("worker_name") or f"__row_{index}")
        grouped.setdefault(worker_name, []).append((index, record))

    prepared: list[dict] = []
    for grouped_worker_name, worker_records in grouped.items():
        worker_half_life = (
            max(1.0, float(half_life_minutes))
            if half_life_minutes is not None
            else recovery_half_life_for_worker(grouped_worker_name)
        )
        ordered = sorted(
            worker_records,
            key=lambda item: (
                _as_datetime(item[1].get("started_at")) or datetime.min,
                _as_datetime(item[1].get("completed_at")) or datetime.min,
            ),
        )
        prior: list[Mapping] = []
        for source_index, record in ordered:
            raw_borg = record.get("borg_cr10")
            try:
                borg = None if raw_borg is None else float(raw_borg)
            except (TypeError, ValueError):
                borg = None
            features = record.get("features")
            if borg is None or not features:
                prior.append(record)
                continue
            started_at = _as_datetime(record.get("started_at"))
            has_prior_same_day = bool(
                started_at
                and any(
                    (completed := _as_datetime(previous.get("completed_at"))) is not None
                    and completed <= started_at
                    and completed.date() == started_at.date()
                    and _effective_borg(previous) is not None
                    for previous in prior
                )
            )
            prepared.append({
                "source_index": source_index,
                "features": with_recovery_features(
                    features,
                    prior,
                    started_at,
                    worker_half_life,
                ),
                "borg_cr10": borg,
                "has_prior_same_day": has_prior_same_day,
            })
            prior.append(record)
    return prepared


def calibrate_recovery_half_life(records: Iterable[Mapping]) -> dict:
    """동일 날짜의 연속 출동이 3건 이상일 때 회복 반감기를 제한적으로 보정한다."""
    source = list(records)
    best: Optional[dict] = None
    for half_life in RECOVERY_HALF_LIFE_CANDIDATES:
        rows = _recovery_training_rows(source, half_life)
        followup_source_indexes = [
            row["source_index"] for row in rows if row["has_prior_same_day"]
        ]
        if len(followup_source_indexes) < MIN_RECOVERY_CALIBRATION_TRANSITIONS:
            continue
        errors: list[float] = []
        rows_by_source = {row["source_index"]: row for row in rows}
        for held_out in followup_source_indexes:
            masked_records = []
            for source_index, record in enumerate(source):
                if source_index != held_out:
                    masked_records.append(record)
                    continue
                masked = dict(record)
                masked["borg_cr10"] = None
                masked["predicted_borg_cr10"] = None
                masked_records.append(masked)
            masked_rows = _recovery_training_rows(masked_records, half_life)
            training = [
                (row["features"], float(row["borg_cr10"]))
                for row in masked_rows
            ]
            parameters = fit_ridge_parameters(training, RECOVERY_FEATURE_KEYS)
            if parameters is None:
                continue
            validation_row = rows_by_source[held_out]
            prediction = predict_with_parameters(parameters, validation_row["features"])
            if prediction is not None:
                errors.append(abs(prediction - float(validation_row["borg_cr10"])))
        if len(errors) != len(followup_source_indexes):
            continue
        mae = float(np.mean(errors))
        # 적은 연속 출동으로 극단값을 택하지 않도록 120분에서 멀수록 작은 패널티를 준다.
        score = mae + 0.1 * abs(
            math.log2(half_life / DEFAULT_RECOVERY_HALF_LIFE_MINUTES)
        )
        candidate = {
            "half_life_minutes": half_life,
            "transition_count": len(followup_source_indexes),
            "validation_mae": round(mae, 4),
            "selection_score": round(score, 4),
            "source": "same_day_leave_one_out",
        }
        if best is None or candidate["selection_score"] < best["selection_score"]:
            best = candidate
    return best or {
        "half_life_minutes": DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
        "transition_count": 0,
        "validation_mae": None,
        "selection_score": None,
        "source": "default_insufficient_same_day_transitions",
    }


def fit_recovery_aware_parameters(
    records: Iterable[Mapping],
    half_life_minutes: Optional[float] = None,
    worker_name: Optional[str] = None,
) -> Optional[dict]:
    source = list(records)
    source_worker_names = {
        str(record.get("worker_name") or "").strip()
        for record in source
        if str(record.get("worker_name") or "").strip()
    }
    resolved_worker_name = str(worker_name or "").strip()
    if not resolved_worker_name and len(source_worker_names) == 1:
        resolved_worker_name = next(iter(source_worker_names))
    configured_half_life = WORKER_RECOVERY_HALF_LIFE_MINUTES.get(resolved_worker_name)

    if half_life_minutes is not None:
        row_half_life: Optional[float] = max(1.0, float(half_life_minutes))
        calibration = {
            "half_life_minutes": row_half_life,
            "transition_count": 0,
            "validation_mae": None,
            "selection_score": None,
            "source": "configured",
        }
    elif configured_half_life is not None:
        row_half_life = configured_half_life
        calibration = {
            "half_life_minutes": configured_half_life,
            "transition_count": 0,
            "validation_mae": None,
            "selection_score": None,
            "source": "configured_worker_order",
        }
    elif len(source_worker_names) > 1:
        row_half_life = None
        calibration = {
            "half_life_minutes": DEFAULT_RECOVERY_HALF_LIFE_MINUTES,
            "half_life_minutes_by_worker": dict(WORKER_RECOVERY_HALF_LIFE_MINUTES),
            "transition_count": 0,
            "validation_mae": None,
            "selection_score": None,
            "source": "configured_worker_order_by_worker",
        }
    else:
        calibration = calibrate_recovery_half_life(source)
        row_half_life = calibration["half_life_minutes"]

    rows = _recovery_training_rows(source, row_half_life)
    training = [(row["features"], row["borg_cr10"]) for row in rows]
    parameters = fit_ridge_parameters(training, RECOVERY_FEATURE_KEYS)
    if parameters is None:
        return None
    parameters["recovery"] = calibration
    return parameters


def recovery_half_life_from_parameters(
    parameters: Optional[Mapping],
    worker_name: Optional[str],
) -> float:
    """저장 모델 메타데이터와 고정 작업자 설정에서 적용 반감기를 찾는다."""
    recovery = (parameters or {}).get("recovery") or {}
    by_worker = recovery.get("half_life_minutes_by_worker") or {}
    normalized_name = str(worker_name or "").strip()
    fallback = by_worker.get(
        normalized_name,
        recovery.get("half_life_minutes", DEFAULT_RECOVERY_HALF_LIFE_MINUTES),
    )
    return recovery_half_life_for_worker(normalized_name, fallback)


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
    current_started_at=None,
    worker_name: Optional[str] = None,
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
    normalized_worker_name = str(worker_name or "").strip()
    if not normalized_worker_name:
        normalized_worker_name = next(
            (
                str(row.get("worker_name") or "").strip()
                for row in reversed(ordered)
                if str(row.get("worker_name") or "").strip()
            ),
            "",
        )
    personal_parameters = fit_recovery_aware_parameters(
        ordered,
        worker_name=normalized_worker_name,
    )
    personal_half_life = recovery_half_life_from_parameters(
        personal_parameters,
        normalized_worker_name,
    )
    personal_features = with_recovery_features(
        current_features,
        ordered,
        current_started_at,
        personal_half_life,
    )
    prediction = predict_with_parameters(personal_parameters, personal_features)
    prediction_source = "personal_actual" if prediction is not None else None
    model_version = RECOVERY_MODEL_VERSION
    applied_half_life = personal_half_life
    applied_features = personal_features
    if prediction is None:
        fallback_half_life = recovery_half_life_from_parameters(
            fallback_parameters,
            normalized_worker_name,
        )
        fallback_features = with_recovery_features(
            current_features,
            ordered,
            current_started_at,
            fallback_half_life,
        )
        prediction = predict_with_parameters(fallback_parameters, fallback_features)
        if prediction is not None:
            prediction_source = "stored_initial"
            model_version = fallback_model_version or MODEL_VERSION
            applied_half_life = fallback_half_life
            applied_features = fallback_features
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
        "cumulative_fatigue": applied_features.get(CUMULATIVE_FATIGUE_FEATURE_KEY, 0.0),
        "recovered_daily_load": applied_features.get(CUMULATIVE_FATIGUE_FEATURE_KEY, 0.0),
        "recovery_minutes": applied_features.get(RECOVERY_MINUTES_FEATURE_KEY, 0.0),
        "recovery_half_life_minutes": applied_half_life,
        "recovery_rate_per_minute": round(recovery_rate_per_minute(applied_half_life), 8),
        "readiness_rule": {
            "minimum_actual_responses": READY_RESPONSE_COUNT,
            "minimum_validations": READY_VALIDATION_COUNT,
            "maximum_recent_mae": READY_MAE_THRESHOLD,
        },
    }
