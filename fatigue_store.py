"""피로 모델 파라미터의 DB 저장·조회와 테스트 초기 모델 시드."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from fatigue_model import (
    FEATURE_KEYS,
    RECOVERY_FEATURE_KEYS,
    RECOVERY_MODEL_VERSION,
    fit_recovery_aware_parameters,
    fit_ridge_parameters,
)
from models import FatigueModel, User


SEED_PATH = Path(__file__).resolve().parent / "test" / "fatigue_seed.json"
SEED_MODEL_VERSION = "experience-seed-ridge-v1"
DATASET_MODEL_VERSION = "dataset-recovery-ridge-v3"


def _seed_payload() -> dict:
    with SEED_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def seed_initial_fatigue_models(
    db: Session,
    organization_id: int,
    include_worker_baselines: bool = True,
) -> None:
    """경험 기반 유사 데이터로 공통 모델과 작업자 초기 참고값을 1회 생성한다."""
    payload = _seed_payload()
    samples = payload.get("samples") or []
    training = [
        ({key: float(row.get(key, 0) or 0) for key in FEATURE_KEYS}, float(row["borg_cr10"]))
        for row in samples
        if row.get("borg_cr10") is not None
    ]
    parameters = fit_ridge_parameters(training)
    baselines = payload.get("worker_baselines") or []
    baseline_values = [float(row["baseline_borg_cr10"]) for row in baselines]
    global_baseline = round(sum(baseline_values) / len(baseline_values), 1) if baseline_values else None

    global_model = db.query(FatigueModel).filter(
        FatigueModel.organization_id == organization_id,
        FatigueModel.scope == "global",
        FatigueModel.is_active.is_(True),
    ).first()
    if global_model is None:
        global_model = FatigueModel(
            organization_id=organization_id,
            scope="global",
            model_version=SEED_MODEL_VERSION,
            feature_schema=list(FEATURE_KEYS),
            parameters=parameters,
            baseline_borg_cr10=global_baseline,
            source="test_seed",
            training_sample_count=len(training),
            actual_response_count=0,
            validation_count=0,
            is_active=True,
        )
        db.add(global_model)

    admin_names = {
        name for name, in db.query(User.full_name).filter(
            User.organization_id == organization_id,
            User.role == "admin",
        ).all()
    }
    users_by_name = {
        user.full_name: user
        for user in db.query(User).filter(
            User.organization_id == organization_id,
            User.role == "worker",
        ).all()
    }
    for item in baselines if include_worker_baselines else []:
        worker_name = str(item.get("worker_name") or "").strip()
        if not worker_name or worker_name in admin_names:
            continue
        existing = db.query(FatigueModel).filter(
            FatigueModel.organization_id == organization_id,
            FatigueModel.scope == "personal",
            FatigueModel.worker_name == worker_name,
            FatigueModel.is_active.is_(True),
        ).first()
        if existing is not None:
            continue
        worker_user = users_by_name.get(worker_name)
        db.add(FatigueModel(
            organization_id=organization_id,
            user_id=worker_user.id if worker_user else None,
            worker_name=worker_name,
            scope="personal",
            model_version=SEED_MODEL_VERSION,
            feature_schema=list(FEATURE_KEYS),
            parameters=parameters,
            baseline_borg_cr10=float(item["baseline_borg_cr10"]),
            source="test_seed",
            training_sample_count=len(training),
            actual_response_count=0,
            validation_count=0,
            is_active=True,
        ))
    db.commit()


def find_active_fatigue_model(
    db: Session,
    organization_id: int,
    user_id: Optional[int],
    worker_name: Optional[str],
) -> Optional[FatigueModel]:
    """개인 모델을 우선하고 없으면 조직 공통 모델을 반환한다."""
    personal_filters = []
    if user_id is not None:
        personal_filters.append(FatigueModel.user_id == user_id)
    if worker_name:
        personal_filters.append(and_(
            FatigueModel.user_id.is_(None),
            FatigueModel.worker_name == worker_name,
        ))
    if personal_filters:
        personal = db.query(FatigueModel).filter(
            FatigueModel.organization_id == organization_id,
            FatigueModel.scope == "personal",
            FatigueModel.is_active.is_(True),
            or_(*personal_filters),
        ).order_by(FatigueModel.user_id.desc(), FatigueModel.updated_at.desc()).first()
        if personal is not None:
            return personal
    return db.query(FatigueModel).filter(
        FatigueModel.organization_id == organization_id,
        FatigueModel.scope == "global",
        FatigueModel.is_active.is_(True),
    ).order_by(FatigueModel.updated_at.desc()).first()


def train_fatigue_models_from_dataset(
    db: Session,
    organization_id: int,
    samples: list[dict],
    *,
    commit: bool = True,
) -> dict:
    """검증된 업로드 행으로 조직 공통 모델과 작업자별 모델을 즉시 교체한다.

    업로드 목표값은 실제 서비스 설문 응답이 아니므로 ``actual_response_count``에는
    포함하지 않는다. 따라서 예측은 바로 가능하지만 기존 설문 생략 기준은 유지된다.
    """
    global_training = [
        (sample["features"], float(sample["borg_cr10"])) for sample in samples
    ]
    global_parameters = fit_recovery_aware_parameters(samples)
    if global_parameters is None:
        raise ValueError("조직 공통 모델 학습에 필요한 데이터가 부족합니다.")
    global_provenance: dict[str, int] = {}
    for sample in samples:
        label = str(sample.get("data_provenance") or "uploaded_dataset")
        global_provenance[label] = global_provenance.get(label, 0) + 1
    global_parameters["training_data_provenance"] = global_provenance

    db.query(FatigueModel).filter(
        FatigueModel.organization_id == organization_id,
        FatigueModel.scope == "global",
        FatigueModel.is_active.is_(True),
    ).update({FatigueModel.is_active: False}, synchronize_session=False)

    global_model = FatigueModel(
        organization_id=organization_id,
        scope="global",
        model_version=DATASET_MODEL_VERSION,
        feature_schema=list(RECOVERY_FEATURE_KEYS),
        parameters=global_parameters,
        baseline_borg_cr10=round(
            sum(borg for _, borg in global_training) / len(global_training), 1
        ),
        source="dataset_import",
        training_sample_count=len(global_training),
        actual_response_count=0,
        validation_count=0,
        is_active=True,
        trained_at=datetime.now(),
    )
    db.add(global_model)

    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        worker_name = str(sample.get("worker_name") or "").strip()
        if worker_name:
            grouped.setdefault(worker_name, []).append(sample)

    users_by_name = {
        user.full_name.strip(): user
        for user in db.query(User).filter(
            User.organization_id == organization_id,
            User.role == "worker",
        ).all()
        if user.full_name.strip()
    }
    admin_names = {
        name.strip() for name, in db.query(User.full_name).filter(
            User.organization_id == organization_id,
            User.role == "admin",
        ).all()
        if name and name.strip()
    }

    trained_personal: list[dict] = []
    skipped_personal: list[dict] = []
    for worker_name, worker_samples in sorted(grouped.items()):
        if worker_name in admin_names:
            skipped_personal.append({
                "worker_name": worker_name,
                "sample_count": len(worker_samples),
                "reason": "admin_account",
            })
            continue
        if len(worker_samples) < 3:
            skipped_personal.append({
                "worker_name": worker_name,
                "sample_count": len(worker_samples),
                "reason": "minimum_3_samples_required",
            })
            continue

        training = [
            (sample["features"], float(sample["borg_cr10"]))
            for sample in worker_samples
        ]
        parameters = fit_recovery_aware_parameters(
            worker_samples,
            worker_name=worker_name,
        )
        if parameters is None:
            continue
        provenance: dict[str, int] = {}
        for sample in worker_samples:
            label = str(sample.get("data_provenance") or "uploaded_dataset")
            provenance[label] = provenance.get(label, 0) + 1
        parameters["training_data_provenance"] = provenance
        worker_user = users_by_name.get(worker_name)
        personal_match = [FatigueModel.worker_name == worker_name]
        if worker_user is not None:
            personal_match.append(FatigueModel.user_id == worker_user.id)
        db.query(FatigueModel).filter(
            FatigueModel.organization_id == organization_id,
            FatigueModel.scope == "personal",
            FatigueModel.is_active.is_(True),
            or_(*personal_match),
        ).update({FatigueModel.is_active: False}, synchronize_session=False)

        personal_model = FatigueModel(
            organization_id=organization_id,
            user_id=worker_user.id if worker_user else None,
            worker_name=worker_name,
            scope="personal",
            model_version=DATASET_MODEL_VERSION,
            feature_schema=list(RECOVERY_FEATURE_KEYS),
            parameters=parameters,
            baseline_borg_cr10=round(
                sum(borg for _, borg in training) / len(training), 1
            ),
            source="dataset_import",
            training_sample_count=len(training),
            actual_response_count=0,
            validation_count=0,
            is_active=True,
            trained_at=datetime.now(),
        )
        db.add(personal_model)
        trained_personal.append({
            "worker_name": worker_name,
            "sample_count": len(training),
            "linked_user_id": worker_user.id if worker_user else None,
            "recovery": parameters.get("recovery"),
            "training_data_provenance": provenance,
        })

    if commit:
        db.commit()
        db.refresh(global_model)
    else:
        db.flush()
    return {
        "global_model_id": global_model.id,
        "global_sample_count": len(global_training),
        "personal_model_count": len(trained_personal),
        "personal_models": trained_personal,
        "skipped_personal_models": skipped_personal,
        "model_version": DATASET_MODEL_VERSION,
    }


def save_personal_fatigue_model(
    db: Session,
    user: User,
    history: Iterable[dict],
    readiness: dict,
) -> Optional[FatigueModel]:
    """실제 응답이 3회 이상이면 개인 모델을 갱신해 재학습 비용을 줄인다."""
    source_history = list(history)
    training = [
        (row["features"], float(row["borg_cr10"]))
        for row in source_history
        if row.get("features") and row.get("borg_cr10") is not None
    ]
    parameters = fit_recovery_aware_parameters(
        source_history,
        worker_name=user.full_name,
    )
    if parameters is None:
        return None

    model = db.query(FatigueModel).filter(
        FatigueModel.organization_id == user.organization_id,
        FatigueModel.scope == "personal",
        or_(
            FatigueModel.user_id == user.id,
            and_(FatigueModel.user_id.is_(None), FatigueModel.worker_name == user.full_name),
        ),
        FatigueModel.is_active.is_(True),
    ).order_by(FatigueModel.user_id.desc()).first()
    if model is None:
        model = FatigueModel(
            organization_id=user.organization_id,
            user_id=user.id,
            worker_name=user.full_name,
            scope="personal",
            model_version=RECOVERY_MODEL_VERSION,
            feature_schema=list(RECOVERY_FEATURE_KEYS),
            source="operational",
            is_active=True,
        )
        db.add(model)

    model.user_id = user.id
    model.worker_name = user.full_name
    model.model_version = RECOVERY_MODEL_VERSION
    model.feature_schema = list(RECOVERY_FEATURE_KEYS)
    model.parameters = parameters
    model.baseline_borg_cr10 = training[-1][1]
    model.source = "operational"
    model.training_sample_count = len(training)
    model.actual_response_count = readiness.get("actual_response_count", len(training))
    model.validation_count = readiness.get("validation_count", 0)
    model.validation_mae = readiness.get("validation_mae")
    model.trained_at = datetime.now()
    db.commit()
    db.refresh(model)
    return model
