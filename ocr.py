"""
비전 LLM(OpenRouter / Gemma 등) 기반 신청서 파서.

신청서 이미지를 멀티모달 모델에 넘겨, 아래 형태의 구조화 JSON을 바로 받는다:
  {"header": {신청번호, 신청일자, 신청부서, 신청자, 연락처},
   "items":  [{품명, 설치장소, 수량, 필요인원수}]}

Template OCR과 달리 고정 영역 지정이 없어, 표 위치·레이아웃이 조금 달라져도 모델이
스스로 읽어 구조화한다. 필요인원수는 여기서 1로 두고 api.py 가 products 마스터로 채운다.

.env:
  OPENROUTER_API_KEY  : OpenRouter API 키 (sk-or-...)
  OCR_MODEL           : 사용할 비전 모델 ID (기본 google/gemma-3-27b-it) — 반드시 이미지 입력 지원 모델

extract_application() 반환 형태는 이전(Textract/CLOVA) 버전과 동일하므로 api.py 는 무변경.
"""
import os
import re
import json
import base64

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OCR_TIMEOUT = 60  # 초 (VLM은 응답이 느릴 수 있음)

# 표준 헤더 키 (모델이 반환해야 할 필드)
HEADER_KEYS = ("신청번호", "신청일자", "신청부서", "신청자", "연락처")

EXTRACT_PROMPT = """다음은 물품 신청서 이미지다. 내용을 읽어 아래 JSON 형식으로만 답하라.
설명 문장이나 코드블록(```) 없이 순수 JSON만 출력한다.

{
  "header": {
    "신청번호": "",
    "신청일자": "",
    "신청부서": "",
    "신청자": "",
    "연락처": ""
  },
  "items": [
    {"품명": "", "설치장소": "", "수량": 0}
  ]
}

규칙:
- 값이 안 보이면 빈 문자열("")로 둔다. 없는 값을 지어내지 않는다.
- 품목 표의 각 행을 items 배열의 원소로 만든다.
- 수량은 정수로 적는다.
- 합계/소계 같은 요약 행은 제외한다.
- 신청일자는 가능하면 YYYY-MM-DD 형식으로 정규화한다.
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


def _call_vlm(file_bytes):
    """OpenRouter(OpenAI 호환)로 이미지+프롬프트를 보내 모델 응답 텍스트를 반환."""
    mime = _detect_mime(file_bytes)
    b64 = base64.b64encode(file_bytes).decode("ascii")
    payload = {
        "model": _get_model(),
        "temperature": 0,  # 환각 최소화
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
            "X-Title": "gjt-demo-server OCR",
        },
        data=json.dumps(payload),
        timeout=OCR_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text):
    """모델 응답 텍스트에서 첫 JSON 오브젝트를 추출·파싱한다 (코드블록/설명 섞여도 대응)."""
    if not text:
        raise ValueError("모델 응답이 비어 있습니다.")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON을 찾지 못했습니다: {text[:200]}")
    return json.loads(match.group(0))


def extract_application(file_bytes):
    """
    신청서 이미지 바이트를 받아 구조화 결과를 반환.

    반환: {"header": {...}, "items": [{품명, 설치장소, 수량, 필요인원수}, ...]}
    """
    content = _call_vlm(file_bytes)
    parsed = _extract_json(content)

    raw_header = parsed.get("header") or {}
    header = {
        k: (str(raw_header[k]).strip() if raw_header.get(k) is not None else "")
        for k in HEADER_KEYS if k in raw_header
    }

    items = []
    for it in (parsed.get("items") or []):
        name = str(it.get("품명") or "").strip()
        if not name:  # 품명 없는 행(빈 행/합계 등) 제외
            continue
        items.append({
            "품명":       name,
            "설치장소":   str(it.get("설치장소") or "").strip(),
            "수량":       _to_int(it.get("수량")),
            "필요인원수": 1,  # products 마스터에서 채워짐 (api.py._enrich_items_with_master)
        })

    return {"header": header, "items": items}
