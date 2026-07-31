import os
import re
import json
import base64
from collections.abc import Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OCR_TIMEOUT = 60  # 초 (VLM은 응답이 느릴 수 있음)
OCR_MAX_IMAGES = int(os.getenv("OCR_MAX_IMAGES", "10"))

# 표준 헤더 키 (모델이 반환해야 할 필드)
HEADER_KEYS = ("신청번호", "신청일자", "신청부서", "신청자", "연락처")

EXTRACT_PROMPT = """다음 이미지는 한 건의 물품 신청서를 페이지별 또는 영역별로 촬영한 것이다.
이미지가 여러 장이면 모두 하나의 신청서로 보고, 입력된 순서대로 함께 읽어라. 한 장에 헤더가 있고 다른 장에 품목 표가 있을 수 있다.
겹쳐 촬영된 영역이나 같은 품목이 여러 이미지에 반복되어도 items에는 한 번만 넣는다.

문서를 읽어 아래 JSON 형식으로만 답하라.
설명 문장이나 코드블록 없이 순수 JSON만 출력한다.

[출력 형식]
{
  "header": {"신청번호": "", "신청일자": "", "신청부서": "", "신청자": "", "연락처": ""},
  "items": [{"자산번호": "", "품명": "", "규격모델": "", "설치장소": "", "수량": 0, "금액": ""}]
}

[헤더 필드 읽기 — 라벨과 같은 행 또는 칸의 값을 우선해 정확히 읽어라]
- 신청번호: '신청번호' 라벨의 값. 없을 때만 접수번호·문서번호를 사용한다. 품목의 자산번호·관리번호와 혼동하지 않는다.
- 신청일자: '신청일자' 또는 '신청일' 라벨의 값. 없을 때만 작성일·접수일을 사용한다. 가능하면 YYYY-MM-DD.
- 신청부서: '신청부서'·'소속부서'·'부서' 라벨의 값. 수신부서·처리부서가 아닌 신청한 부서를 읽는다.
- 신청자: '신청자'·'성명'·'작성자' 라벨의 값. 결재자·담당자·수령자와 혼동하지 않는다.
- 연락처: '연락처'·'전화번호'·'휴대폰' 라벨의 값. 다른 사람의 전화번호가 있으면 신청자 칸과 연결된 번호를 읽는다.
- 자산번호: 품목 표의 각 행에 적힌 자산관리번호
- 품명: 물품 이름
- 규격모델: 규격 또는 모델명
- 설치장소: 설치·보관 위치
- 수량: 개수(정수)
- 금액: 금액(적힌 그대로, 예 '1,000,000원')

[규칙]
- 문서에 실제로 적힌 글자만 읽는다. 위 예시 값을 그대로 복사하거나 임의 값을 지어내지 않는다.
- 신청번호는 매 문서마다 실제로 보이는 값을 다시 읽고, 없으면 반드시 빈 문자열("").
- 값이 안 보이는 필드는 빈 문자열("")로 둔다.
- 품목 표의 각 행을 items 원소로 만들고, 합계/소계 같은 요약 행은 제외한다.
"""

# 품목 표를 읽는 프롬프트와 분리해 헤더만 다시 읽으면, 작은 글씨·표 테두리 때문에
# 누락되기 쉬운 헤더 필드의 재현율을 높일 수 있다. 메인 결과에서 빈 필드가 있을 때만 호출한다.
HEADER_REVIEW_PROMPT = """다음 이미지는 한 건의 물품 신청서다. 여러 장이면 모두 같은 신청서의 페이지 또는 확대 사진이므로 함께 읽어라.
품목 표는 무시하고, 문서의 헤더/기본정보 영역만 확대해서 재확인하라. 각 값은 반드시 해당 라벨과 같은 행 또는 칸에서 읽는다.

아래 JSON 형식으로만 답하라. 설명, 마크다운, 코드블록은 넣지 마라.
{
  "header": {"신청번호": "", "신청일자": "", "신청부서": "", "신청자": "", "연락처": ""}
}

우선순위:
- 신청번호: '신청번호', 없을 때만 접수번호·문서번호. 자산번호·품목번호는 절대 사용하지 않는다.
- 신청일자: '신청일자' 또는 '신청일', 없을 때만 작성일·접수일.
- 신청부서: '신청부서'·'소속부서'·'부서'. 수신/처리 부서는 사용하지 않는다.
- 신청자: '신청자'·'성명'·'작성자'. 결재자/담당자는 사용하지 않는다.
- 연락처: 신청자와 연결된 '연락처'·'전화번호'·'휴대폰'.

읽을 수 없는 값은 추측하지 말고 빈 문자열로 둔다. 날짜는 가능하면 YYYY-MM-DD로 쓴다.
"""


def _to_int(value, default=1):
    digits = re.sub(r"[^\d]", "", str(value if value is not None else ""))
    return int(digits) if digits else default


def _get_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 없음: .env 에 OpenRouter 키를 넣으세요.")
    return key


def _get_model():
    return os.getenv("OCR_MODEL", "google/gemma-3-27b-it")


def _detect_mime(file_bytes):
    """파일 시그니처로 data URI mime 타입을 추정한다."""
    if file_bytes[:4] == b"\x89PNG":
        return "image/png"
    if file_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if file_bytes[:4] == b"%PDF":
        return "application/pdf"  # 주: 모델에 따라 PDF 미지원일 수 있음
    if file_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "image/jpeg"


def _as_file_list(file_bytes_or_list) -> list[bytes]:
    """단일 이미지와 이미지 목록을 모두 받되, 모델 호출 전에는 항상 목록으로 통일한다."""
    if isinstance(file_bytes_or_list, (bytes, bytearray)):
        files = [bytes(file_bytes_or_list)]
    elif isinstance(file_bytes_or_list, Sequence):
        files = list(file_bytes_or_list)
    else:
        raise TypeError("이미지 바이트 또는 이미지 바이트 목록이 필요합니다.")

    if not files:
        raise ValueError("처리할 이미지가 없습니다.")
    if len(files) > OCR_MAX_IMAGES:
        raise ValueError(f"한 신청서에는 최대 {OCR_MAX_IMAGES}장까지 업로드할 수 있습니다.")
    if any(not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes for file_bytes in files):
        raise ValueError("비어 있거나 올바르지 않은 이미지가 포함되어 있습니다.")
    return [bytes(file_bytes) for file_bytes in files]


def _call_vlm(file_bytes_or_list, prompt=EXTRACT_PROMPT, max_tokens=4096):
    """OpenRouter(OpenAI 호환)로 한 신청서의 이미지들을 함께 보내고 모델 응답을 반환한다."""
    file_bytes_list = _as_file_list(file_bytes_or_list)
    content = [{"type": "text", "text": prompt}]
    for file_bytes in file_bytes_list:
        mime = _detect_mime(file_bytes)
        b64 = base64.b64encode(file_bytes).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    payload = {
        "model": _get_model(),
        "temperature": 0,      # 환각 최소화
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": content,
        }],
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
            "X-Title": "gjt-demo-server OCR",
        },
        json=payload,
        timeout=OCR_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text):
    """
    모델 응답 텍스트에서 JSON 오브젝트를 추출·파싱한다.
    - 코드블록/설명 문장이 섞여도 첫 '{'부터 균형 잡힌 '}'까지만 잡는다(문자열 내 중괄호 무시).
    - 응답이 잘려 균형이 안 맞으면 '잘림'을 명확히 알린다.
    """
    if not text:
        raise ValueError("모델 응답이 비어 있습니다.")
    t = re.sub(r"```(?:json)?", "", text)  # 코드펜스 제거

    start = t.find("{")
    if start == -1:
        raise ValueError(f"응답에서 JSON을 찾지 못했습니다: {text[:300]}")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])

    # 끝까지 depth가 0으로 안 돌아옴 → 응답이 중간에 잘림
    raise ValueError(f"JSON이 완결되지 않았습니다(응답 잘림 가능). 원본 끝부분: ...{text[-300:]}")


def _header_from_parsed(parsed) -> dict:
    """응답의 header를 고정된 키 집합으로 정리한다."""
    if not isinstance(parsed, dict):
        raise ValueError("모델 응답 JSON의 최상위 값은 객체여야 합니다.")
    raw_header = parsed.get("header") or {}
    if not isinstance(raw_header, dict):
        raw_header = {}
    return {
        key: str(raw_header.get(key) or "").strip()
        for key in HEADER_KEYS
    }


def _review_missing_header(file_bytes_list, header: dict) -> dict:
    """메인 OCR에서 누락된 헤더만 전용 프롬프트로 재검증해 채운다."""
    missing_keys = [key for key in HEADER_KEYS if not header.get(key)]
    if not missing_keys:
        return header

    # 이 호출은 품질 보완용이다. 이미 확보한 품목 OCR 결과를 재검증 실패 때문에 버리지 않는다.
    try:
        reviewed = _header_from_parsed(
            _extract_json(_call_vlm(file_bytes_list, HEADER_REVIEW_PROMPT, max_tokens=1024))
        )
    except Exception:
        return header

    return {
        key: header.get(key) or reviewed.get(key, "")
        for key in HEADER_KEYS
    }


def extract_application(file_bytes_or_list):
    """
    신청서 이미지 한 장 또는 같은 신청서에 속한 이미지 목록을 받아 구조화 결과를 반환.
    여러 장은 입력 순서대로 하나의 Vision LLM 요청에 넣어, 헤더와 품목을 한 신청서로 합친다.

    반환: {"header": {...}, "items": [{품명, 설치장소, 수량, 필요인원수}, ...]}
    """
    file_bytes_list = _as_file_list(file_bytes_or_list)
    content = _call_vlm(file_bytes_list)
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("모델 응답 JSON의 최상위 값은 객체여야 합니다.")

    header = _review_missing_header(file_bytes_list, _header_from_parsed(parsed))

    items = []
    for it in (parsed.get("items") or []):
        if not isinstance(it, dict):
            continue
        name = str(it.get("품명") or "").strip()
        if not name:  # 품명 없는 행(빈 행/합계 등) 제외
            continue
        items.append({
            "자산번호":   str(it.get("자산번호") or "").strip(),
            "품명":       name,
            "규격모델":   str(it.get("규격모델") or "").strip(),
            "설치장소":   str(it.get("설치장소") or "").strip(),
            "수량":       _to_int(it.get("수량")),
            "금액":       str(it.get("금액") or "").strip(),
            "필요인원수": 1,  # products 마스터에서 채워짐 (api.py._enrich_items_with_master)
        })

    return {"header": header, "items": items}
