"""관리자가 업로드한 피로도 데이터셋을 현재 모델 피처로 변환한다."""

from __future__ import annotations

import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fatigue_model import (
    CUMULATIVE_FATIGUE_FEATURE_KEY,
    FEATURE_KEYS,
    LEGACY_RECOVERY_FEATURE_KEY,
    RECOVERY_FEATURE_KEYS,
    RECOVERY_MINUTES_FEATURE_KEY,
    recovery_half_life_for_worker,
    with_recovery_features,
)


MAX_DATASET_BYTES = 5 * 1024 * 1024
MIN_DATASET_SAMPLES = 3


class FatigueDatasetError(ValueError):
    """업로드 파일 형식이나 행 데이터가 학습에 적합하지 않을 때 발생한다."""


COLUMN_ALIASES = {
    "worker_name": (
        "worker_name", "worker", "name", "작업자", "작업자명", "근로자", "근로자명", "근로학생", "이름",
    ),
    "work_minutes": (
        "work_minutes", "working_minutes", "work_time", "작업시간", "작업분", "실작업시간",
    ),
    "total_minutes": (
        "total_minutes", "elapsed_minutes", "total_time", "전체시간", "총시간", "경과시간",
    ),
    "driving_minutes": (
        "driving_minutes", "drive_minutes", "driving_time", "차량이동시간", "운전시간", "이동시간",
    ),
    "unknown_minutes": (
        "unknown_minutes", "unclassified_minutes", "unknown_time", "미분류시간", "판단불가시간",
    ),
    "item_count": (
        "item_count", "items", "schedule_count", "물품개수", "일정개수", "개수",
    ),
    "labor_load": (
        "labor_load", "workload", "required_people_sum", "노동량", "부하량", "필요인원합계",
    ),
    "team_size": (
        "team_size", "actual_team_size", "workers_count", "투입인원수", "작업인원수", "팀인원",
    ),
    "borg_cr10": (
        "borg_cr10", "borg10", "borg", "fatigue", "expected_fatigue", "피로도", "예상피로도",
    ),
    "started_at": (
        "started_at", "work_started_at", "start_time", "작업시작시각", "시작시각", "출동일시",
    ),
    "completed_at": (
        "completed_at", "work_completed_at", "end_time", "작업완료시각", "완료시각", "측정시점",
    ),
    "cumulative_fatigue": (
        "cumulative_fatigue", "recovered_daily_load", "누적피로도", "누적 피로도",
    ),
    "recovery_minutes": (
        "recovery_minutes", "rest_minutes", "휴식시간", "휴식시간(분)", "회복시간", "회복시간(분)",
    ),
}

EASYPICKUP_SHEET = "01_모델입력"
EASYPICKUP_RECORDED_RESPONSE_WORKERS = {"최현", "김경언"}
EASYPICKUP_REQUIRED_COLUMNS = {
    "신청번호",
    "근로학생",
    "작업참여여부(0/1)",
    "측정단계",
    "측정시점",
    "경과시간(분)",
    "개수(개)",
    "노동량(필요인원×왕복횟수)",
    "Borg CR10",
    "최종출동ID",
    "출동일시",
    "계획작업시간(분)",
}
EASYPICKUP_OPTIONAL_MODEL_COLUMNS = {"누적피로도", "휴식시간(분)"}


def _normalize_column(value: Any) -> str:
    return re.sub(r"[\s_\-()/]+", "", str(value).strip().lower())


_ALIAS_TO_COLUMN = {
    _normalize_column(alias): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def dataset_template() -> dict:
    """프론트와 Swagger가 동일하게 사용할 업로드 스키마를 반환한다."""
    return {
        "supported_extensions": [".csv", ".json", ".xlsx"],
        "time_unit": "minutes",
        "minimum_total_samples": MIN_DATASET_SAMPLES,
        "minimum_personal_samples": MIN_DATASET_SAMPLES,
        "target": "borg_cr10",
        "optional_identity_column": "worker_name",
        "feature_columns": list(RECOVERY_FEATURE_KEYS),
        "required_columns": [
            "work_minutes", "item_count", "labor_load", "team_size", "borg_cr10",
        ],
        "defaults": {
            "driving_minutes": 0,
            "unknown_minutes": 0,
            "total_minutes": "work_minutes + driving_minutes + unknown_minutes",
        },
        "optional_context_columns": ["started_at", "completed_at"],
        "derived_feature_rules": {
            "cumulative_fatigue": "같은 날 이전 출동 Borg의 개인 반감기 감쇠 합. 첫 출동은 0",
            "recovery_minutes": "같은 날 직전 출동 완료부터 현재 출동 시작까지의 분. 첫 출동은 0",
        },
        "specialized_xlsx_sheet": EASYPICKUP_SHEET,
        "column_aliases": {key: list(value) for key, value in COLUMN_ALIASES.items()},
    }


def _enrich_samples_with_recovery(samples: list[dict]) -> dict:
    """샘플을 시간순으로 정렬해 누적피로도와 회복시간을 모델 피처에 추가한다."""
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, sample in enumerate(samples):
        worker_name = str(sample.get("worker_name") or "").strip()
        group_key = worker_name or f"__row_{index}"
        grouped.setdefault(group_key, []).append((index, sample))

    provided_rows = 0
    validated_rows = 0
    for worker_name, worker_samples in grouped.items():
        half_life = recovery_half_life_for_worker(worker_name)
        ordered = sorted(
            worker_samples,
            key=lambda item: (
                item[1].get("started_at") or datetime.min,
                item[1].get("completed_at") or datetime.min,
            ),
        )
        prior: list[dict] = []
        for _, sample in ordered:
            declared_cumulative = sample.pop("_declared_cumulative_fatigue", None)
            declared_recovery = sample.pop("_declared_recovery_minutes", None)
            calculated = with_recovery_features(
                sample.get("features") or {},
                prior,
                sample.get("started_at"),
                half_life,
            )
            sample["features"] = calculated
            if declared_cumulative is not None or declared_recovery is not None:
                provided_rows += 1
                dispatch_label = sample.get("dispatch_id") or "행"
                if declared_cumulative is not None and abs(
                    float(declared_cumulative)
                    - float(calculated[CUMULATIVE_FATIGUE_FEATURE_KEY])
                ) > 0.0011:
                    raise FatigueDatasetError(
                        f"{worker_name}의 출동 {dispatch_label} 누적피로도가 계산 규칙과 다릅니다."
                    )
                if declared_recovery is not None and abs(
                    float(declared_recovery)
                    - float(calculated[RECOVERY_MINUTES_FEATURE_KEY])
                ) > 0.01:
                    raise FatigueDatasetError(
                        f"{worker_name}의 출동 {dispatch_label} 휴식시간이 계산 규칙과 다릅니다."
                    )
                validated_rows += 1
            prior.append(sample)
    return {"provided_rows": provided_rows, "validated_rows": validated_rows}


def _read_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
            except (pd.errors.ParserError, ValueError) as exc:
                raise FatigueDatasetError("CSV의 열 또는 행 형식이 올바르지 않습니다.") from exc
        raise FatigueDatasetError("CSV 인코딩은 UTF-8 또는 CP949여야 합니다.") from last_error
    if suffix == ".xlsx":
        try:
            workbook = pd.ExcelFile(io.BytesIO(content))
            sheet_name = (
                EASYPICKUP_SHEET
                if EASYPICKUP_SHEET in workbook.sheet_names
                else workbook.sheet_names[0]
            )
            return pd.read_excel(workbook, sheet_name=sheet_name)
        except Exception as exc:
            raise FatigueDatasetError("XLSX 파일을 읽을 수 없습니다.") from exc
    if suffix == ".json":
        try:
            payload = json.loads(content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FatigueDatasetError("JSON 파일 형식이 올바르지 않습니다.") from exc
        if isinstance(payload, dict):
            payload = payload.get("samples")
        if not isinstance(payload, list):
            raise FatigueDatasetError('JSON은 행 배열 또는 {"samples": [...]} 형식이어야 합니다.')
        try:
            return pd.DataFrame(payload)
        except (TypeError, ValueError) as exc:
            raise FatigueDatasetError("JSON samples의 각 항목은 같은 형태의 객체여야 합니다.") from exc
    raise FatigueDatasetError("지원 형식은 CSV, JSON, XLSX입니다.")


def _datetime_value(value: Any, *, row_number: int, column: str):
    if value is None or pd.isna(value):
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FatigueDatasetError(
            f"{row_number}행의 {column} 값은 날짜와 시각이어야 합니다."
        ) from exc
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed


def _parse_easypickup_model_input(frame: pd.DataFrame) -> dict:
    """이지픽업 행을 실제 서비스의 작업자별 출동 세션 단위로 집계한다."""
    missing = EASYPICKUP_REQUIRED_COLUMNS.difference(str(column) for column in frame.columns)
    if missing:
        raise FatigueDatasetError(
            f"{EASYPICKUP_SHEET} 시트의 필수 열이 없습니다: " + ", ".join(sorted(missing))
        )

    source = frame.copy()
    participant = pd.to_numeric(source["작업참여여부(0/1)"], errors="coerce") == 1
    post_work = source["측정단계"].astype(str).str.strip() == "작업후"
    has_borg = pd.to_numeric(source["Borg CR10"], errors="coerce").notna()
    source = source.loc[participant & post_work & has_borg].copy()
    if source.empty:
        raise FatigueDatasetError("참여한 작업자의 작업후 Borg 기록이 없습니다.")

    numeric_columns = (
        "경과시간(분)",
        "개수(개)",
        "노동량(필요인원×왕복횟수)",
        "Borg CR10",
        "계획작업시간(분)",
    )
    for column in numeric_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    invalid = source[list(numeric_columns)].isna().any(axis=1)
    if invalid.any():
        first_index = int(source.index[invalid][0]) + 2
        raise FatigueDatasetError(f"{first_index}행의 학습용 숫자 값이 비어 있거나 잘못되었습니다.")

    source["근로학생"] = source["근로학생"].astype(str).str.strip()
    if (source["근로학생"] == "").any():
        first_index = int(source.index[source["근로학생"] == ""][0]) + 2
        raise FatigueDatasetError(f"{first_index}행의 근로학생 이름이 비어 있습니다.")
    source["출동일시"] = pd.to_datetime(source["출동일시"], errors="coerce")
    source["측정시점"] = pd.to_datetime(source["측정시점"], errors="coerce")
    invalid_time = source[["출동일시", "측정시점"]].isna().any(axis=1)
    if invalid_time.any():
        first_index = int(source.index[invalid_time][0]) + 2
        raise FatigueDatasetError(f"{first_index}행의 출동일시 또는 측정시점이 잘못되었습니다.")

    dispatch_team_sizes = source.groupby("최종출동ID")["근로학생"].nunique().to_dict()
    samples: list[dict] = []
    grouped = source.groupby(["근로학생", "최종출동ID"], sort=True, dropna=False)
    for (worker_name, dispatch_id), rows in grouped:
        started_at = rows["출동일시"].min().to_pydatetime()
        completed_at = rows["측정시점"].max().to_pydatetime()
        work_minutes = float(rows["경과시간(분)"].max())
        total_minutes = max(work_minutes, float(rows["계획작업시간(분)"].max()))
        item_count = float(rows["개수(개)"].sum())
        labor_load = float(rows["노동량(필요인원×왕복횟수)"].sum())
        borg = float(rows["Borg CR10"].max())
        team_size = float(dispatch_team_sizes.get(dispatch_id, 1))
        values = {
            "work_minutes": work_minutes,
            "total_minutes": total_minutes,
            "driving_minutes": 0.0,
            "unknown_minutes": 0.0,
            "item_count": item_count,
            "labor_load": labor_load,
            "team_size": team_size,
        }
        if any(value < 0 for value in values.values()) or not 0 <= borg <= 10:
            raise FatigueDatasetError(
                f"{worker_name}의 출동 {dispatch_id} 데이터 범위가 올바르지 않습니다."
            )
        declared_cumulative = None
        declared_recovery = None
        if "누적피로도" in rows.columns:
            provided = pd.to_numeric(rows["누적피로도"], errors="coerce").dropna()
            if not provided.empty:
                declared_cumulative = float(provided.max())
        if "휴식시간(분)" in rows.columns:
            provided = pd.to_numeric(rows["휴식시간(분)"], errors="coerce").dropna()
            if not provided.empty:
                declared_recovery = float(provided.max())
        samples.append({
            "worker_name": worker_name,
            "dispatch_id": str(dispatch_id),
            "data_provenance": (
                "recorded_response"
                if worker_name in EASYPICKUP_RECORDED_RESPONSE_WORKERS
                else "synthetic_assumption"
            ),
            "started_at": started_at,
            "completed_at": completed_at,
            "features": {key: values[key] for key in FEATURE_KEYS},
            "borg_cr10": borg,
            "_declared_cumulative_fatigue": declared_cumulative,
            "_declared_recovery_minutes": declared_recovery,
        })

    if len(samples) < MIN_DATASET_SAMPLES:
        raise FatigueDatasetError(
            f"학습에는 최소 {MIN_DATASET_SAMPLES}개의 출동 표본이 필요합니다."
        )
    derived_validation = _enrich_samples_with_recovery(samples)
    recognized_columns = EASYPICKUP_REQUIRED_COLUMNS.union(
        EASYPICKUP_OPTIONAL_MODEL_COLUMNS.intersection(str(column) for column in frame.columns)
    )
    return {
        "samples": samples,
        "row_count": len(samples),
        "source_row_count": len(source),
        "recognized_columns": sorted(recognized_columns),
        "ignored_columns": [
            str(column) for column in frame.columns
            if str(column) not in recognized_columns
        ],
        "dataset_format": "easypickup_cr10_dispatch",
        "aggregation": "worker_name + 최종출동ID; grouped Borg uses maximum; recovery resets daily",
        "derived_feature_validation": derived_validation,
    }


def _number(value: Any, *, row_number: int, column: str) -> float:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        raise FatigueDatasetError(f"{row_number}행의 {column} 값이 비어 있습니다.")
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FatigueDatasetError(f"{row_number}행의 {column} 값은 숫자여야 합니다.") from exc
    if not math.isfinite(number):
        raise FatigueDatasetError(f"{row_number}행의 {column} 값은 유한한 숫자여야 합니다.")
    return number


def parse_fatigue_dataset(filename: str, content: bytes) -> dict:
    """파일을 읽고 기본 7개 및 회복 2개 피처와 Borg 목표값을 검증한다."""
    if not content:
        raise FatigueDatasetError("빈 데이터셋 파일입니다.")
    if len(content) > MAX_DATASET_BYTES:
        raise FatigueDatasetError("데이터셋 파일은 최대 5MB까지 업로드할 수 있습니다.")

    frame = _read_dataframe(filename, content)
    if frame.empty:
        raise FatigueDatasetError("학습할 데이터 행이 없습니다.")
    if EASYPICKUP_REQUIRED_COLUMNS.issubset(str(column) for column in frame.columns):
        return _parse_easypickup_model_input(frame)

    mapped: dict[str, str] = {}
    ignored_columns: list[str] = []
    for original in frame.columns:
        canonical = _ALIAS_TO_COLUMN.get(_normalize_column(original))
        if canonical is None:
            ignored_columns.append(str(original))
            continue
        if canonical in mapped:
            raise FatigueDatasetError(
                f"{mapped[canonical]}와 {original} 열이 모두 {canonical}(으)로 인식됩니다. 하나만 남겨 주세요."
            )
        mapped[canonical] = str(original)

    required = {"work_minutes", "item_count", "labor_load", "team_size", "borg_cr10"}
    missing = [column for column in required if column not in mapped]
    if missing:
        raise FatigueDatasetError("필수 열이 없습니다: " + ", ".join(sorted(missing)))

    samples: list[dict] = []
    for index, row in frame.iterrows():
        row_number = int(index) + 2
        values: dict[str, float] = {}
        for column in ("work_minutes", "item_count", "labor_load", "team_size"):
            values[column] = _number(row[mapped[column]], row_number=row_number, column=column)
        for column in ("driving_minutes", "unknown_minutes"):
            values[column] = (
                _number(row[mapped[column]], row_number=row_number, column=column)
                if column in mapped and not pd.isna(row[mapped[column]])
                else 0.0
            )
        values["total_minutes"] = (
            _number(row[mapped["total_minutes"]], row_number=row_number, column="total_minutes")
            if "total_minutes" in mapped and not pd.isna(row[mapped["total_minutes"]])
            else values["work_minutes"] + values["driving_minutes"] + values["unknown_minutes"]
        )
        borg = _number(row[mapped["borg_cr10"]], row_number=row_number, column="borg_cr10")

        negative = [key for key, value in values.items() if value < 0]
        if negative:
            raise FatigueDatasetError(f"{row_number}행의 {negative[0]} 값은 0 이상이어야 합니다.")
        if not 0 <= borg <= 10:
            raise FatigueDatasetError(f"{row_number}행의 borg_cr10 값은 0~10이어야 합니다.")
        if values["team_size"] < 1:
            raise FatigueDatasetError(f"{row_number}행의 team_size 값은 1 이상이어야 합니다.")
        for integer_column in ("item_count", "team_size"):
            if not values[integer_column].is_integer():
                raise FatigueDatasetError(
                    f"{row_number}행의 {integer_column} 값은 정수여야 합니다."
                )
        if values["total_minutes"] < max(
            values["work_minutes"], values["driving_minutes"], values["unknown_minutes"]
        ):
            raise FatigueDatasetError(
                f"{row_number}행의 total_minutes는 개별 시간 값보다 작을 수 없습니다."
            )

        worker_name = None
        if "worker_name" in mapped:
            raw_name = row[mapped["worker_name"]]
            if raw_name is not None and not pd.isna(raw_name):
                worker_name = str(raw_name).strip() or None
        started_at = (
            _datetime_value(
                row[mapped["started_at"]], row_number=row_number, column="started_at"
            )
            if "started_at" in mapped
            else None
        )
        completed_at = (
            _datetime_value(
                row[mapped["completed_at"]], row_number=row_number, column="completed_at"
            )
            if "completed_at" in mapped
            else None
        )
        if started_at and completed_at and completed_at < started_at:
            raise FatigueDatasetError(
                f"{row_number}행의 completed_at은 started_at보다 빠를 수 없습니다."
            )
        features = {key: values[key] for key in FEATURE_KEYS}
        if CUMULATIVE_FATIGUE_FEATURE_KEY in mapped:
            features[CUMULATIVE_FATIGUE_FEATURE_KEY] = _number(
                row[mapped[CUMULATIVE_FATIGUE_FEATURE_KEY]],
                row_number=row_number,
                column=CUMULATIVE_FATIGUE_FEATURE_KEY,
            )
        elif LEGACY_RECOVERY_FEATURE_KEY in mapped:
            features[CUMULATIVE_FATIGUE_FEATURE_KEY] = _number(
                row[mapped[LEGACY_RECOVERY_FEATURE_KEY]],
                row_number=row_number,
                column=LEGACY_RECOVERY_FEATURE_KEY,
            )
        if RECOVERY_MINUTES_FEATURE_KEY in mapped:
            features[RECOVERY_MINUTES_FEATURE_KEY] = _number(
                row[mapped[RECOVERY_MINUTES_FEATURE_KEY]],
                row_number=row_number,
                column=RECOVERY_MINUTES_FEATURE_KEY,
            )
        samples.append({
            "worker_name": worker_name,
            "data_provenance": "uploaded_dataset",
            "started_at": started_at,
            "completed_at": completed_at,
            "features": features,
            "borg_cr10": borg,
        })

    if len(samples) < MIN_DATASET_SAMPLES:
        raise FatigueDatasetError(
            f"학습에는 최소 {MIN_DATASET_SAMPLES}개의 데이터 행이 필요합니다."
        )
    derived_validation = _enrich_samples_with_recovery(samples)
    return {
        "samples": samples,
        "row_count": len(samples),
        "recognized_columns": sorted(mapped),
        "ignored_columns": ignored_columns,
        "dataset_format": "generic",
        "derived_feature_validation": derived_validation,
    }
