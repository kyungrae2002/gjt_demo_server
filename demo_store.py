"""시연용 고정 OCR fixture와 로컬 SQLite 초기 데이터 관리.

DEMO_MODE에서는 OCR 결과만 이 fixture에서 읽는다. 신청서·제품·일정의 조회와
변경은 일반 서버와 같은 SQLAlchemy 로직으로 ``test/demo.sqlite3``에 반영한다.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from models import Application, Product


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


DEMO_MODE = _env_flag("DEMO_MODE")
FIXTURE_PATH = Path(__file__).resolve().parent / "test" / "demo_fixture.json"


def _load_fixture() -> dict:
    """fixture와 그것이 참조하는 원본 사진이 모두 있는지 확인해 읽는다."""
    try:
        with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"시연 데이터 파일이 없습니다: {FIXTURE_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"시연 데이터 JSON 형식이 올바르지 않습니다: {FIXTURE_PATH}") from exc

    for filename in fixture.get("source_images", []):
        path = FIXTURE_PATH.parent / filename
        if not path.is_file():
            raise RuntimeError(f"시연 원본 이미지가 없습니다: {path}")
    if not fixture.get("application", {}).get("items"):
        raise RuntimeError("시연 데이터에 OCR 품목 결과가 없습니다.")
    return fixture


def load_demo_ocr_result() -> dict:
    """외부 OCR 호출 대신 저장해 둔 실제 OCR 결과를 복사해 돌려준다."""
    return copy.deepcopy(_load_fixture()["application"])


def load_demo_source_images() -> list[str]:
    """시연 OCR 응답에 기록할 fixture 원본 이미지 이름."""
    return list(_load_fixture().get("source_images", []))


def seed_demo_data(
    db: Session,
    enrich_items: Callable[[list, Session], list],
    unique_application_no: Callable[[Session, str | None], str],
) -> tuple[Application, bool]:
    """비어 있는 시연 DB에 fixture 한 묶음을 한 번만 심는다.

    서버 재시작 때에는 이미 저장된 같은 marker 신청서를 그대로 사용하므로, 사용자가
    API로 변경한 값·일정·제품은 ``test/demo.sqlite3``에 계속 남는다.
    """
    fixture = _load_fixture()
    marker = fixture.get("source_marker") or "[DEMO] fixture"
    existing = db.query(Application).filter(Application.원본파일명 == marker).first()
    if existing:
        return existing, False

    for product in fixture.get("products", []):
        name = str(product.get("품명") or "").strip()
        if not name or db.query(Product).filter(Product.품명 == name).first():
            continue
        db.add(Product(품명=name, 필요인원수=int(product.get("필요인원수") or 1)))
    db.flush()

    parsed = fixture["application"]
    header = parsed.get("header") or {}
    items = enrich_items(parsed.get("items") or [], db)
    first_place = items[0].get("설치장소", "") if items else ""
    row = Application(
        신청번호=unique_application_no(db, header.get("신청번호")),
        신청일자=header.get("신청일자") or "",
        신청부서=header.get("신청부서") or (first_place.split()[0] if first_place else "창고"),
        신청자=header.get("신청자"),
        연락처=header.get("연락처"),
        원본파일명=marker,
        물품목록=items,
        상태="접수",
        점검완료=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, True
