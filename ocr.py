"""
Amazon Textract 기반 신청서 OCR 파서.

신청서 이미지(PNG/JPEG) 또는 단일 페이지 PDF에서
  - 품목 표(TABLES) -> [{품명, 설치장소, 수량, 필요인원수}, ...]
  - 헤더 양식(FORMS) -> {신청번호, 신청일자, 신청부서}
를 추출한다.

Textract 무료 티어(가입 후 3개월): AnalyzeDocument 월 100페이지 무료.
동기 호출(analyze_document, Bytes 직접 전달)은 단일 페이지 이미지/PDF만 지원한다.
멀티 페이지 PDF는 S3 업로드 후 StartDocumentAnalysis(비동기)를 써야 한다.
"""
import os
import re
import boto3

# 표 컬럼 헤더 -> 표준 필드명 (정규화된 텍스트로 매칭)
COLUMN_SYNONYMS = {
    "품명": ["품명", "자산명", "물품명", "품목", "물품", "제품명"],
    "설치장소": ["설치장소", "설치위치", "위치사용명", "위치", "장소", "설치처"],
    "수량": ["수량", "개수", "qty", "수량개수"],
    "필요인원수": ["필요인원수", "필요인원", "인원수", "인원"],
}

# 헤더 양식 key -> 표준 필드명
HEADER_SYNONYMS = {
    "신청번호": ["신청번호", "신청서번호", "접수번호", "문서번호"],
    "신청일자": ["신청일자", "접수일자", "신청일", "일자", "작성일"],
    "신청부서": ["신청부서", "신청조직", "부서", "신청기관", "소속"],
}


def _normalize(text):
    """공백/특수문자 제거 후 소문자화 — 헤더 매칭용."""
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


def _get_textract_client():
    return boto3.client(
        "textract",
        region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _block_text(block, blocks_by_id):
    """CELL/KEY 블록의 자식 WORD·SELECTION 텍스트를 이어붙인다."""
    words = []
    for rel in block.get("Relationships", []):
        if rel["Type"] != "CHILD":
            continue
        for cid in rel["Ids"]:
            child = blocks_by_id.get(cid, {})
            if child.get("BlockType") == "WORD":
                words.append(child.get("Text", ""))
            elif child.get("BlockType") == "SELECTION_ELEMENT":
                if child.get("SelectionStatus") == "SELECTED":
                    words.append("[X]")
    return " ".join(words).strip()


def _parse_tables(blocks, blocks_by_id):
    """모든 TABLE 블록에서 품목 행을 추출한다."""
    items = []
    for tbl in [b for b in blocks if b["BlockType"] == "TABLE"]:
        # 셀 좌표 -> 텍스트 격자 구성
        grid = {}
        max_row = max_col = 0
        for rel in tbl.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for cid in rel["Ids"]:
                cell = blocks_by_id.get(cid, {})
                if cell.get("BlockType") != "CELL":
                    continue
                r, c = cell["RowIndex"], cell["ColumnIndex"]
                grid[(r, c)] = _block_text(cell, blocks_by_id)
                max_row, max_col = max(max_row, r), max(max_col, c)

        if max_row < 2:  # 헤더 + 데이터 최소 2행 필요
            continue

        # 1행(헤더)으로 컬럼 인덱스 -> 표준 필드 매핑
        col_to_field = {}
        for c in range(1, max_col + 1):
            field = _match_field(grid.get((1, c), ""), COLUMN_SYNONYMS)
            if field:
                col_to_field[c] = field

        # 품명 컬럼조차 못 찾으면 이 표는 품목 표가 아님
        if "품명" not in col_to_field.values():
            continue

        for r in range(2, max_row + 1):
            row = {"품명": "", "설치장소": "", "수량": 1, "필요인원수": 1}
            for c, field in col_to_field.items():
                val = grid.get((r, c), "").strip()
                if field in ("수량", "필요인원수"):
                    row[field] = _to_int(val, default=1)
                else:
                    row[field] = val
            if row["품명"]:  # 품명 없는 행(빈 행/합계 행)은 건너뜀
                items.append(row)

    return items


def _parse_header(blocks, blocks_by_id):
    """KEY_VALUE_SET(FORMS)에서 신청번호/신청일자/신청부서를 추출한다."""
    header = {}
    for kv in [b for b in blocks if b["BlockType"] == "KEY_VALUE_SET"]:
        if "KEY" not in kv.get("EntityTypes", []):
            continue
        key_text = _block_text(kv, blocks_by_id)
        field = _match_field(key_text, HEADER_SYNONYMS)
        if not field or field in header:
            continue
        # KEY -> VALUE 관계 추적
        value_text = ""
        for rel in kv.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for vid in rel["Ids"]:
                    value_text = _block_text(blocks_by_id.get(vid, {}), blocks_by_id)
                    break
        if value_text:
            header[field] = value_text
    return header


def extract_application(file_bytes):
    """
    신청서 이미지/단일페이지 PDF 바이트를 받아 구조화 결과를 반환.

    반환: {"header": {...}, "items": [{품명, 설치장소, 수량, 필요인원수}, ...]}
    """
    client = _get_textract_client()
    resp = client.analyze_document(
        Document={"Bytes": file_bytes},
        FeatureTypes=["TABLES", "FORMS"],
    )
    blocks = resp["Blocks"]
    blocks_by_id = {b["Id"]: b for b in blocks}

    return {
        "header": _parse_header(blocks, blocks_by_id),
        "items": _parse_tables(blocks, blocks_by_id),
    }
