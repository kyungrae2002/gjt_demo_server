"""지정 조직의 작업자 계정과 엑셀 기반 개인 피로 모델을 함께 준비한다."""

from __future__ import annotations

import secrets
from typing import Iterable, Mapping

from sqlalchemy.orm import Session

from auth import _hash_password, _normalize_email, _normalize_username, _validate_password
from fatigue_store import train_fatigue_models_from_dataset
from models import Organization, User


TARGET_WORKER_NAMES = ("최현", "김경언", "강경래", "박우민", "정하람")
DEFAULT_ACCOUNT_SPECS = (
    {"full_name": "최현", "username": "choihyun", "email": "choihyun@gjt.local"},
    {"full_name": "김경언", "username": "kimkyungeon", "email": "kimkyungeon@gjt.local"},
    {"full_name": "강경래", "username": "kangkyungrae", "email": "kangkyungrae@gjt.local"},
    {"full_name": "박우민", "username": "parkwumin", "email": "parkwumin@gjt.local"},
    {"full_name": "정하람", "username": "jeongharam", "email": "jeongharam@gjt.local"},
)


def _generated_password() -> str:
    return f"Gjt-{secrets.token_urlsafe(12)}-{secrets.randbelow(10)}"


def _validated_specs(account_specs: Iterable[Mapping]) -> list[dict]:
    specs: list[dict] = []
    for raw in account_specs:
        full_name = str(raw.get("full_name") or "").strip()
        if not full_name:
            raise ValueError("모든 계정에 full_name이 필요합니다.")
        username = _normalize_username(str(raw.get("username") or ""))
        email = _normalize_email(str(raw.get("email") or ""))
        supplied_password = raw.get("password")
        password = (
            _validate_password(str(supplied_password))
            if supplied_password is not None
            else _generated_password()
        )
        specs.append({
            "full_name": full_name,
            "username": username,
            "email": email,
            "password": password,
            "password_generated": supplied_password is None,
        })

    names = [item["full_name"] for item in specs]
    if len(names) != len(set(names)):
        raise ValueError("계정 설정에 같은 full_name이 두 번 이상 있습니다.")
    if set(names) != set(TARGET_WORKER_NAMES):
        raise ValueError(
            "계정 설정은 다음 5명을 정확히 포함해야 합니다: "
            + ", ".join(TARGET_WORKER_NAMES)
        )
    usernames = [item["username"] for item in specs]
    emails = [item["email"] for item in specs]
    if len(usernames) != len(set(usernames)) or len(emails) != len(set(emails)):
        raise ValueError("아이디와 이메일은 계정마다 달라야 합니다.")
    return specs


def provision_personalized_workers(
    db: Session,
    organization: Organization,
    samples: list[dict],
    account_specs: Iterable[Mapping] = DEFAULT_ACCOUNT_SPECS,
) -> dict:
    """5개 계정을 멱등적으로 준비하고 같은 트랜잭션에서 개인 모델을 연결한다."""
    specs = _validated_specs(account_specs)
    sample_names = {
        str(sample.get("worker_name") or "").strip()
        for sample in samples
        if sample.get("worker_name")
    }
    missing_samples = set(TARGET_WORKER_NAMES).difference(sample_names)
    if missing_samples:
        raise ValueError(
            "개인화 데이터가 없는 작업자가 있습니다: " + ", ".join(sorted(missing_samples))
        )

    accounts: list[dict] = []
    try:
        for spec in specs:
            matches = db.query(User).filter(
                User.organization_id == organization.id,
                User.role == "worker",
                User.full_name == spec["full_name"],
            ).all()
            if len(matches) > 1:
                raise ValueError(
                    f"{spec['full_name']} 이름의 기존 작업자 계정이 여러 개라 자동 연결할 수 없습니다."
                )
            if matches:
                user = matches[0]
                accounts.append({
                    "id": user.id,
                    "full_name": user.full_name,
                    "username": user.username,
                    "email": user.email,
                    "status": "existing",
                    "initial_password": None,
                })
                continue

            username_owner = db.query(User).filter(User.username == spec["username"]).first()
            if username_owner is not None:
                raise ValueError(f"이미 사용 중인 아이디입니다: {spec['username']}")
            email_owner = db.query(User).filter(User.email == spec["email"]).first()
            if email_owner is not None:
                raise ValueError(f"이미 사용 중인 이메일입니다: {spec['email']}")

            user = User(
                organization_id=organization.id,
                role="worker",
                username=spec["username"],
                email=spec["email"],
                full_name=spec["full_name"],
                password_hash=_hash_password(spec["password"]),
                is_active=True,
            )
            db.add(user)
            db.flush()
            accounts.append({
                "id": user.id,
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
                "status": "created",
                "initial_password": spec["password"],
                "password_generated": spec["password_generated"],
            })

        training = train_fatigue_models_from_dataset(
            db,
            organization.id,
            samples,
            commit=False,
        )
        trained_names = {
            item["worker_name"] for item in training["personal_models"]
        }
        missing_models = set(TARGET_WORKER_NAMES).difference(trained_names)
        if missing_models:
            raise ValueError(
                "개인 모델 생성에 실패한 작업자가 있습니다: "
                + ", ".join(sorted(missing_models))
            )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "organization_id": organization.id,
        "organization_name": organization.name,
        "accounts": accounts,
        "training": training,
    }
