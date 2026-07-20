"""
NAVER CLOVA OCR (Template) 기반 신청서 파서.

신청서 이미지(PNG/JPEG/TIFF) 또는 PDF를 CLOVA OCR로 인식해
  - 헤더 필드(fields)  -> {신청번호, 신청일자, 신청부서, 신청자, 연락처}
  - 품목 표(tables)    -> [{품명, 설치장소, 수량, 필요인원수}]
를 추출한다.

전제:
  - NCP CLOVA OCR 콘솔에서 신청서 양식을 Template 로 등록하고,
    헤더 필드와 표(table) 영역을 지정해 둔다.
  - .env 에 Clova_Invoke_URL, Clova_Secret_Key 를 넣어 둔다.

extract_application() 반환 형태 {"header":..., "items":...} 는 Textract 버전과 동일하므로
api.py 는 수정할 필요가 없다.
"""
import os
import re
import json
import time
import uuid
import base64

import requests
from dotenv import load_dotenv

load_dotenv()

# 표 컬럼 헤더/템플릿 필드명 -> 표준 필드명 (정규화된 텍스트로 매칭)
COLUMN_SYNONYMS = {
    "품명": ["품명", "자산명", "물품명", "품목", "물품", "제품명"],
    "설치장소": ["설치장소", "설치위치", "위치사용명", "위치", "장소", "설치처"],
    "수량": ["수량", "개수", "qty", "수량개수"],
    "필요인원수": ["필요인원수", "필요인원", "인원수", "인원"],
}

# 헤더(템플릿 필드) 이름 -> 표준 필드명
HEADER_SYNONYMS = {
    "신청번호": ["신청번호", "신청서번호", "접수번호", "문서번호"],
    "신청일자": ["신청일자", "접수일자", "신청일", "일자", "작성일"],
    "신청부서": ["신청부서", "신청조직", "부서", "신청기관", "소속"],
    "신청자":   ["신청자", "신청인", "담당자", "작성자", "성명"],
    "연락처":   ["연락처", "전화번호", "휴대폰", "핸드폰", "전화", "tel", "hp"],
}

CLOVA_TIMEOUT = 30  # 초


def _normalize(text):
    """공백/특수문자 제거 후 소문자화 — 필드/헤더 매칭용."""
    return re.sub(r"[\s:()\[\]/·.,-]", "", (text or "")).lower()


def _match_field(text, synonyms_map):
    """정규화된 text가 어떤 표준 필드에 해당하는지 반환 (없으면 None)."""
    norm = _normalize(text)
    if not norm:
        return None
    for field, syns in synonyms_map.items():
        for syn in syns:
            if _normalize(syn) in norm or norm in _normalize(syn):
                return field
    return None


def _to_int(text, default=1):
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else default


def _get_clova_config():
    """CLOVA OCR 호출 설정(.env). 이름 대소문자 변형도 허용."""
    url = os.getenv("Clova_Invoke_URL") or os.getenv("CLOVA_INVOKE_URL")
    secret = os.getenv("Clova_Secret_Key") or os.getenv("CLOVA_SECRET_KEY")
    if not url or not secret:
        raise RuntimeError(
            "CLOVA OCR 설정 없음: .env 에 Clova_Invoke_URL / Clova_Secret_Key 를 넣으세요."
        )
    # CLOVA OCR는 HTTPS(443)만 받는다. http:// 로 저장돼 있으면 보정한다.
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url, secret


def _detect_format(file_bytes):
    """파일 시그니처로 CLOVA format 값을 추정한다."""
    if file_bytes[:4] == b"\x89PNG":
        return "png"
    if file_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if file_bytes[:4] == b"%PDF":
        return "pdf"
    if file_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "jpg"


def _call_clova(file_bytes):
    """CLOVA OCR API 호출 → 응답 JSON 반환."""
    url, secret = _get_clova_config()
    payload = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [{
            "format": _detect_format(file_bytes),
            "name": "application",
            "data": base64.b64encode(file_bytes).decode("ascii"),
        }],
    }
    resp = requests.post(
        url,
        headers={"X-OCR-SECRET": secret, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=CLOVA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _cell_text(cell):
    """CLOVA 표 셀의 텍스트를 이어붙인다."""
    words = []
    for line in cell.get("cellTextLines", []):
        for w in line.get("cellWords", []):
            words.append(w.get("inferText", ""))
    return " ".join(words).strip()


def _parse_fields(fields):
    """Template 필드(fields) -> 표준 헤더 dict."""
    header = {}
    for f in fields:
        std = _match_field(f.get("name", ""), HEADER_SYNONYMS)
        if not std or std in header:
            continue
        value = (f.get("inferText") or "").strip()
        if value:
            header[std] = value
    return header


def _parse_tables(tables):
    """CLOVA 표(tables) -> 품목 행 리스트."""
    items = []
    for tbl in tables:
        cells = tbl.get("cells", [])
        if not cells:
            continue

        # 셀 좌표(0-based) -> 텍스트 격자
        grid = {}
        max_row = max_col = 0
        for cell in cells:
            r = cell.get("rowIndex", 0)
            c = cell.get("columnIndex", 0)
            grid[(r, c)] = _cell_text(cell)
            max_row, max_col = max(max_row, r), max(max_col, c)

        if max_row < 1:  # 헤더(0행) + 데이터(1행 이상) 필요
            continue

        # 0행(헤더)으로 컬럼 인덱스 -> 표준 필드 매핑
        col_to_field = {}
        for c in range(0, max_col + 1):
            field = _match_field(grid.get((0, c), ""), COLUMN_SYNONYMS)
            if field:
                col_to_field[c] = field

        if "품명" not in col_to_field.values():
            continue  # 품목 표가 아님

        for r in range(1, max_row + 1):
            row = {"품명": "", "설치장소": "", "수량": 1, "필요인원수": 1}
            for c, field in col_to_field.items():
                val = grid.get((r, c), "").strip()
                if field in ("수량", "필요인원수"):
                    row[field] = _to_int(val, default=1)
                else:
                    row[field] = val
            if row["품명"]:
                items.append(row)

    return items


def extract_application(file_bytes):
    """
    신청서 이미지/PDF 바이트를 받아 구조화 결과를 반환.

    반환: {"header": {...}, "items": [{품명, 설치장소, 수량, 필요인원수}, ...]}
    """
    result = _call_clova(file_bytes)
    images = result.get("images", [])
    if not images:
        return {"header": {}, "items": []}

    img = images[0]
    infer = img.get("inferResult")
    if infer and infer != "SUCCESS":
        raise RuntimeError(f"CLOVA OCR 인식 실패: {img.get('message', infer)}")

    return {
        "header": _parse_fields(img.get("fields", [])),
        "items": _parse_tables(img.get("tables", [])),
    }
