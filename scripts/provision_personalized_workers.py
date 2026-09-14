"""5개 작업자 계정과 이지픽업 개인 피로 모델을 대상 DB에 프로비저닝한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from db import SessionLocal
from fatigue_dataset import parse_fatigue_dataset
from models import Organization
from worker_personalization import DEFAULT_ACCOUNT_SPECS, provision_personalized_workers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datas" / "이지픽업_CR10_모델입력_현재최종.xlsx"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5명 작업자 계정과 회복도 반영 개인 모델을 한 트랜잭션에서 생성합니다."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--organization-id", type=int)
    group.add_argument("--entry-code")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--account-config",
        type=Path,
        help="full_name, username, email, 선택적 password를 가진 JSON 배열",
    )
    return parser.parse_args()


def _organization(db, args: argparse.Namespace) -> Organization:
    query = db.query(Organization)
    if args.organization_id is not None:
        organization = query.filter(Organization.id == args.organization_id).first()
    elif args.entry_code:
        organization = query.filter(Organization.entry_code == args.entry_code.strip().upper()).first()
    else:
        rows = query.order_by(Organization.id).all()
        if len(rows) != 1:
            raise ValueError("조직이 하나가 아니므로 --organization-id 또는 --entry-code가 필요합니다.")
        organization = rows[0]
    if organization is None:
        raise ValueError("대상 조직을 찾을 수 없습니다.")
    return organization


def main() -> None:
    args = _arguments()
    dataset_path = args.dataset.expanduser().resolve()
    parsed = parse_fatigue_dataset(dataset_path.name, dataset_path.read_bytes())
    account_specs = DEFAULT_ACCOUNT_SPECS
    if args.account_config:
        account_specs = json.loads(args.account_config.read_text(encoding="utf-8"))
        if not isinstance(account_specs, list):
            raise ValueError("account-config는 JSON 배열이어야 합니다.")

    db = SessionLocal()
    try:
        result = provision_personalized_workers(
            db,
            _organization(db, args),
            parsed["samples"],
            account_specs,
        )
    finally:
        db.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
