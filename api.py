import io
import os
from datetime import date, datetime, timedelta
import yaml
import pandas as pd
import boto3
from fastapi import FastAPI, Depends, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from model import run_optimizer
from model2 import run_optimizer as run_route_optimizer, split_location_to_building_room
from db import get_db, engine, Base
from models import Product, Application, Schedule
from ocr import extract_application

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"[경고] DB 연결 실패, 테이블 생성 건너뜀: {e}")

app = FastAPI(docs_url=None)

# 허용할 프론트엔드 출처 (CORS와 Origin 검증이 함께 사용)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://gwanzae.vercel.app",
    "https://gjtdemoserver-production.up.railway.app",
]

# Origin 검증·API 키를 건너뛸 경로 (브라우저 주소창에서 직접 여는 문서·헬스체크)
ORIGIN_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/openapi.yaml"}

# API 키 인증. .env 의 API_KEY 가 설정된 경우에만 강제한다.
# 요청 헤더 X-API-Key 값이 일치해야 통과. (curl 등 무단 요청 차단용)
API_KEY = os.getenv("API_KEY")
API_KEY_HEADER = "X-API-Key"


@app.middleware("http")
async def verify_origin(request: Request, call_next):
    # 문서·헬스체크는 브라우저에서 직접 열 수 있어야 하므로 검증 제외
    if request.url.path in ORIGIN_EXEMPT_PATHS:
        return await call_next(request)
    # CORS preflight(OPTIONS)는 CORSMiddleware가 처리하도록 통과
    if request.method == "OPTIONS":
        return await call_next(request)
    # 동일 출처 GET·Swagger·비브라우저(서버·curl) 요청은 Origin 헤더가 없다(None).
    # cross-origin 위협 요청은 브라우저가 반드시 Origin을 붙이므로, Origin이 있을 때만 검증한다.
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse(
            status_code=403,
            content={"detail": "허용되지 않은 요청 출처입니다."},
        )

    # API 키 인증 (Origin과 무관하게 동작 → curl 등 무단 요청 차단). 미설정 시 건너뜀.
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "유효한 API 키가 필요합니다."},
        )

    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Swagger UI 에 Authorize(자물쇠) 버튼을 띄워 X-API-Key 를 넣어 테스트할 수 있게 한다.
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title="API Docs", version="1.0.0", routes=app.routes)
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": API_KEY_HEADER}
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


# ==========================================
# Pydantic 스키마
# ==========================================
class OptimizeItem(BaseModel):
    품명: str
    설치장소: str
    수량: int
    필요인원수: int

class OptimizeRequest(BaseModel):
    신청번호: str
    신청일자: str
    신청부서: str
    물품목록: List[OptimizeItem]

class RouteItem(BaseModel):
    품명: str
    설치장소: str
    수량: int
    필요인원수: int

class RouteRequest(BaseModel):
    투입인원수: Optional[int] = None
    신청서: List[RouteItem]

class ProductCreate(BaseModel):
    품명: str
    필요인원수: int

class ProductUpdate(BaseModel):
    품명: Optional[str] = None
    필요인원수: Optional[int] = None

class ImportRequest(BaseModel):
    s3_key: str

class ApplicationItem(BaseModel):
    자산번호: Optional[str] = ""
    품명: str
    규격모델: Optional[str] = ""
    설치장소: Optional[str] = ""
    수량: Optional[int] = 1
    금액: Optional[str] = ""
    필요인원수: int

class ApplicationPatch(BaseModel):
    신청번호: Optional[str] = None
    신청일자: Optional[str] = None
    신청부서: Optional[str] = None
    신청자: Optional[str] = None
    연락처: Optional[str] = None
    물품목록: Optional[List[ApplicationItem]] = None
    점검완료: Optional[bool] = None

class SchedulePatch(BaseModel):
    출동일시: Optional[datetime] = None
    자산번호: Optional[str] = None
    품명: Optional[str] = None
    규격모델: Optional[str] = None
    금액: Optional[str] = None
    설치장소: Optional[str] = None
    신청부서: Optional[str] = None
    수량: Optional[int] = None
    필요인원수: Optional[int] = None
    투입인원수: Optional[int] = None
    가용명단: Optional[str] = None
    출동확정: Optional[bool] = None
    동선: Optional[dict] = None


# ==========================================
# 헬스체크
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================
# OpenAPI YAML 명세서 + 커스텀 Swagger UI
# ==========================================
@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml():
    return Response(
        content=yaml.dump(app.openapi(), allow_unicode=True, sort_keys=False),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=openapi.yaml"},
    )

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    html = get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")
    download_btn = (
        '<a href="/openapi.yaml" download '
        'style="position:fixed;top:14px;right:16px;z-index:9999;'
        'background:#49cc90;color:white;padding:8px 18px;'
        'border-radius:4px;text-decoration:none;font-weight:bold;font-size:14px;">'
        'YAML 다운로드</a>'
    )
    body = html.body.decode().replace("</body>", f"{download_btn}</body>")
    return HTMLResponse(content=body)


# ==========================================
# 최적화
# ==========================================
@app.post("/optimize")
async def optimize(data: List[OptimizeRequest]):
    rows = []
    for req in data:
        for item in req.물품목록:
            rows.append({
                "신청번호":   req.신청번호,
                "신청일자":   req.신청일자,
                "신청부서":   req.신청부서,
                "품명":       item.품명,
                "설치장소":   item.설치장소,
                "수량":       item.수량,
                "필요인원수": item.필요인원수,
            })
    df       = pd.DataFrame(rows)
    df_avail = pd.read_csv("datas/근로학생시간.csv")
    results  = run_optimizer(df, df_avail)
    return JSONResponse(content=results)


# ==========================================
# 신청서 OCR 저장 (Vision LLM / OpenRouter)
# ==========================================
def _normalize_date(raw: Optional[str]) -> str:
    """OCR로 읽은 날짜 문자열을 YYYY-MM-DD로 정규화. 실패 시 오늘 날짜."""
    if raw:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return date.today().strftime("%Y-%m-%d")


def _gen_application_no() -> str:
    """신청번호 미검출 시 자동 생성 (타임스탬프 기반)."""
    return "OCR-" + datetime.now().strftime("%Y%m%d%H%M%S%f")


def _unique_application_no(db: Session, base: Optional[str]) -> str:
    """
    신청번호를 DB에서 유니크하게 보정한다.
    - base가 비었으면 타임스탬프 기반으로 생성.
    - base가 이미 존재하면 -2, -3 … 접미사를 붙여 충돌을 피한다.
    OCR로 읽은 신청번호가 부정확·중복이어도 일괄 처리가 멈추지 않게 한다.
    """
    base = (base or "").strip()
    if not base:
        base = _gen_application_no()
    num, i = base, 2
    while db.query(Application).filter(Application.신청번호 == num).first():
        num = f"{base}-{i}"
        i += 1
    return num


def _enrich_items_with_master(items: list, db: Session) -> list:
    """각 품목의 필요인원수를 products 마스터(품명)로 채운다. 없으면 OCR값→1."""
    enriched = []
    for it in items:
        name = (it.get("품명") or "").strip()
        master = db.query(Product).filter(Product.품명 == name).first()
        ppl = master.필요인원수 if master else int(it.get("필요인원수") or 1)
        enriched.append({
            "자산번호":   (it.get("자산번호") or "").strip(),
            "품명":       name,
            "규격모델":   (it.get("규격모델") or "").strip(),
            "설치장소":   (it.get("설치장소") or "").strip(),
            "수량":       int(it.get("수량") or 1),
            "금액":       (it.get("금액") or "").strip(),
            "필요인원수": int(ppl),
        })
    return enriched


def _application_to_dict(app: Application) -> dict:
    return {
        "id":         app.id,
        "신청번호":   app.신청번호,
        "신청일자":   app.신청일자,
        "신청부서":   app.신청부서,
        "신청자":     app.신청자,
        "연락처":     app.연락처,
        "원본파일명": app.원본파일명,
        "물품목록":   app.물품목록,
        "상태":       app.상태,
        "점검완료":   app.점검완료,
    }


@app.post("/ocr/applications")
async def create_application_from_ocr(
    file: UploadFile = File(...),
    신청번호: Optional[str] = Form(None),
    신청일자: Optional[str] = Form(None),
    신청부서: Optional[str] = Form(None),
    신청자: Optional[str] = Form(None),
    연락처: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    신청서 이미지(PNG/JPEG/TIFF)를 업로드하면
    비전 LLM(OpenRouter)으로 기본정보·품목 표를 추출해 applications 에 저장한다. (상태=접수, 점검완료=false)

    - 헤더 필드(신청번호/신청일자/신청부서/신청자/연락처)는 폼 값으로 넘기면 OCR 결과보다 우선한다.
    - 각 품목 필요인원수는 products 마스터(품명) 값으로 채운다. (없으면 OCR값→1)
    - 이후 [점검] 단계에서 기본정보·인원수를 확인·수정한다.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        parsed = extract_application(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR 처리 실패: {e}")

    items = parsed["items"]
    if not items:
        raise HTTPException(
            status_code=422,
            detail="신청서에서 품목 표를 인식하지 못했습니다. 표 형태/화질을 확인해 주세요.",
        )

    hdr = parsed["header"]
    # 폼 값 > OCR 헤더 순으로 신청번호 후보를 잡고, DB에서 유니크하게 보정(충돌 시 접미사).
    num = _unique_application_no(db, 신청번호 or hdr.get("신청번호"))
    day = _normalize_date(신청일자 or hdr.get("신청일자"))
    first_place = items[0]["설치장소"].strip() if items[0].get("설치장소") else ""
    dept = 신청부서 or hdr.get("신청부서") or (first_place.split()[0] if first_place else "창고")
    applicant = 신청자 or hdr.get("신청자")
    contact = 연락처 or hdr.get("연락처")

    app_row = Application(
        신청번호=num,
        신청일자=day,
        신청부서=dept,
        신청자=applicant,
        연락처=contact,
        원본파일명=file.filename,
        물품목록=_enrich_items_with_master(items, db),
        상태="접수",
        점검완료=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)
    return _application_to_dict(app_row)


# ==========================================
# 신청서 점검 (기본정보 + 품목별 인원수 확인·수정)
# ==========================================
def _upsert_product_master(db: Session, 품명: str, 필요인원수: int):
    """products 마스터를 품명 기준으로 갱신/추가한다 (앞으로 조회부터 적용)."""
    name = (품명 or "").strip()
    if not name:
        return
    prod = db.query(Product).filter(Product.품명 == name).first()
    if prod:
        prod.필요인원수 = int(필요인원수)
    else:
        db.add(Product(품명=name, 필요인원수=int(필요인원수)))


@app.get("/applications/{app_id}")
def get_application(app_id: int, db: Session = Depends(get_db)):
    """점검용 상세 조회(id로 식별): 기본정보 + 품목별 인원수."""
    app_row = db.query(Application).filter(Application.id == app_id).first()
    if not app_row:
        raise HTTPException(status_code=404, detail="신청서를 찾을 수 없습니다.")
    return _application_to_dict(app_row)


@app.patch("/applications/{app_id}")
def update_application(app_id: int, body: ApplicationPatch, db: Session = Depends(get_db)):
    """
    점검 단계: 기본정보(신청번호/신청일자/신청부서/신청자/연락처)와 품목별 인원수를 수정하고
    점검완료 처리한다.

    - 신청번호를 바꾸면 유니크 검사 후 반영한다. (점검 단계엔 아직 일정이 없어 안전)
    - 물품목록의 필요인원수를 수정하면 products 마스터도 upsert(품명 기준) → 이후 신청서에 일반 적용.
      (이미 저장된 다른 신청서 스냅샷은 소급 변경하지 않음)
    """
    app_row = db.query(Application).filter(Application.id == app_id).first()
    if not app_row:
        raise HTTPException(status_code=404, detail="신청서를 찾을 수 없습니다.")

    if body.신청번호 is not None and body.신청번호.strip() != app_row.신청번호:
        new_no = body.신청번호.strip()
        if not new_no:
            raise HTTPException(status_code=400, detail="신청번호는 빈 값일 수 없습니다.")
        if db.query(Application).filter(
            Application.신청번호 == new_no, Application.id != app_id
        ).first():
            raise HTTPException(status_code=409, detail=f"이미 존재하는 신청번호입니다: {new_no}")
        app_row.신청번호 = new_no

    if body.신청일자 is not None:
        app_row.신청일자 = body.신청일자
    if body.신청부서 is not None:
        app_row.신청부서 = body.신청부서
    if body.신청자 is not None:
        app_row.신청자 = body.신청자
    if body.연락처 is not None:
        app_row.연락처 = body.연락처

    if body.물품목록 is not None:
        new_items = []
        for it in body.물품목록:
            new_items.append({
                "자산번호":   (it.자산번호 or "").strip(),
                "품명":       it.품명.strip(),
                "규격모델":   (it.규격모델 or "").strip(),
                "설치장소":   (it.설치장소 or "").strip(),
                "수량":       int(it.수량 or 1),
                "금액":       (it.금액 or "").strip(),
                "필요인원수": int(it.필요인원수),
            })
            # 인원수 수정분을 마스터에 반영 (앞으로 적용)
            _upsert_product_master(db, it.품명, it.필요인원수)
        app_row.물품목록 = new_items

    if body.점검완료 is not None:
        app_row.점검완료 = body.점검완료

    db.commit()
    db.refresh(app_row)
    return _application_to_dict(app_row)


@app.patch("/applications/{app_id}/complete")
def complete_application(app_id: int, db: Session = Depends(get_db)):
    """수거 완료 처리(id로 식별, 상태=완료). 이후 재최적화 대상에서 제외된다."""
    app_row = db.query(Application).filter(Application.id == app_id).first()
    if not app_row:
        raise HTTPException(status_code=404, detail="신청서를 찾을 수 없습니다.")
    app_row.상태 = "완료"
    db.commit()
    db.refresh(app_row)
    return _application_to_dict(app_row)


# ==========================================
# 최적화 실행 (점검완료·미완료 신청서 전체 → 일정 갱신)
# ==========================================
def _parse_dispatch_label(label: str):
    """'2026-07-20 (월) 09:00' 형태 라벨 → datetime. 실패 시 None."""
    try:
        parts = label.split()
        return datetime.strptime(f"{parts[0]} {parts[-1]}", "%Y-%m-%d %H:%M")
    except Exception:
        return None


@app.post("/optimize/run")
def optimize_run(db: Session = Depends(get_db)):
    """
    점검완료(점검완료=true)이고 아직 완료되지 않은 신청서 전체를 합쳐 재최적화한다.
    기존 일정(schedules)은 폐기하고 새로 작성하며, 배정된 신청서는 상태=일정확정으로 바꾼다.
    """
    apps = (
        db.query(Application)
        .filter(Application.점검완료 == True, Application.상태 != "완료")
        .all()
    )
    if not apps:
        raise HTTPException(status_code=400, detail="최적화할 (점검완료·미완료) 신청서가 없습니다.")

    rows = []
    for a in apps:
        for it in (a.물품목록 or []):
            rows.append({
                "신청번호":   a.신청번호,
                "신청일자":   a.신청일자,
                "신청부서":   a.신청부서,
                "자산번호":   it.get("자산번호", ""),
                "품명":       it.get("품명", ""),
                "규격모델":   it.get("규격모델", ""),
                "설치장소":   it.get("설치장소", ""),
                "수량":       int(it.get("수량") or 1),
                "금액":       it.get("금액", ""),
                "필요인원수": int(it.get("필요인원수") or 1),
            })

    df       = pd.DataFrame(rows)
    df_avail = pd.read_csv("datas/근로학생시간.csv")
    try:
        results = run_optimizer(df, df_avail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최적화 실행 오류: {e}")

    신청번호s = [a.신청번호 for a in apps]
    # 기존 일정 폐기 후 재작성 (충돌 해소)
    db.query(Schedule).filter(Schedule.신청번호.in_(신청번호s)).delete(synchronize_session=False)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    for r in results:
        db.add(Schedule(
            신청번호=r["신청번호"],
            출동일시=_parse_dispatch_label(r["출동일시"]),
            자산번호=r.get("자산번호"),
            품명=r["품명"],
            규격모델=r.get("규격모델"),
            금액=r.get("금액"),
            설치장소=r["설치장소"],
            신청부서=r["신청부서"],
            수량=int(r.get("수량") or 1),
            필요인원수=int(r.get("필요인원수") or 1),
            optimize_run_id=run_id,
            출동확정=False,
        ))

    # 실제 배정된 신청서만 일정확정 처리 (슬롯 못 잡은 건 접수 상태 유지)
    scheduled_nums = {r["신청번호"] for r in results}
    for a in apps:
        if a.신청번호 in scheduled_nums:
            a.상태 = "일정확정"

    db.commit()
    return {
        "optimize_run_id": run_id,
        "대상_신청서수":   len(apps),
        "일정확정_신청서": sorted(scheduled_nums),
        "일정_행수":       len(results),
    }


# ==========================================
# 조회 (신청서 목록 / 오늘 출동 일정)
# ==========================================
def _schedule_to_dict(s: Schedule) -> dict:
    return {
        "id":              s.id,
        "신청번호":        s.신청번호,
        "출동일시":        s.출동일시,
        "자산번호":        s.자산번호,
        "품명":            s.품명,
        "규격모델":        s.규격모델,
        "금액":            s.금액,
        "설치장소":        s.설치장소,
        "신청부서":        s.신청부서,
        "수량":            s.수량,
        "필요인원수":      s.필요인원수,
        "투입인원수":      s.투입인원수,
        "가용명단":        s.가용명단,
        "출동확정":        s.출동확정,
        "동선":            s.동선,
        "optimize_run_id": s.optimize_run_id,
    }


@app.get("/applications")
def list_applications(상태: Optional[str] = None, db: Session = Depends(get_db)):
    """신청서 목록을 '처리해야 할 순서'(배정된 출동일시 오름차순)로 반환. 미배정은 뒤로."""
    from sqlalchemy import func as safunc

    q = db.query(Application)
    if 상태:
        q = q.filter(Application.상태 == 상태)
    apps = q.all()

    # 신청번호별 최초 출동일시
    dispatch = dict(
        db.query(Schedule.신청번호, safunc.min(Schedule.출동일시))
        .group_by(Schedule.신청번호)
        .all()
    )

    result = []
    for a in apps:
        d = _application_to_dict(a)
        d["출동일시"] = dispatch.get(a.신청번호)
        result.append(d)

    result.sort(key=lambda x: (x["출동일시"] is None, x["출동일시"] or datetime.max))
    return result


@app.get("/schedules/today")
def schedules_today(db: Session = Depends(get_db)):
    """오늘(출동일시 기준) 수거 일정. 시간대별 렌더링은 프론트에서."""
    today = date.today()
    start = datetime(today.year, today.month, today.day)
    end   = start + timedelta(days=1)
    rows = (
        db.query(Schedule)
        .filter(Schedule.출동일시 >= start, Schedule.출동일시 < end)
        .order_by(Schedule.출동일시)
        .all()
    )
    return [_schedule_to_dict(s) for s in rows]


# ==========================================
# 오늘 출동 확정 (건물별 동선 계산·저장) + 일정 수정
# ==========================================
@app.post("/dispatch/confirm")
def dispatch_confirm(투입인원수: Optional[int] = None, db: Session = Depends(get_db)):
    """
    오늘 출동 일정을 확정한다. 건물별 실내 수거 동선(model2)을 계산해 각 일정에 저장하고
    출동확정=true로 표시한다. 이후 수정이 필요하면 PATCH /schedules/{id}로 갱신한다.
    """
    today = date.today()
    start = datetime(today.year, today.month, today.day)
    end   = start + timedelta(days=1)
    rows = (
        db.query(Schedule)
        .filter(Schedule.출동일시 >= start, Schedule.출동일시 < end)
        .all()
    )
    if not rows:
        raise HTTPException(status_code=400, detail="오늘 출동할 일정이 없습니다.")

    df = pd.DataFrame([{
        "신청번호":   s.신청번호,
        "품명":       s.품명,
        "설치장소":   s.설치장소,
        "수량":       s.수량,
        "필요인원수": s.필요인원수,
    } for s in rows])

    dispatch_people = float(투입인원수) if 투입인원수 is not None else None
    try:
        route_results = run_route_optimizer(df, None, dispatch_people)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"건물 그래프 파일 없음: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동선 계산 오류: {e}")

    route_by_building = {r.get("건물명"): r for r in route_results}
    for s in rows:
        building = split_location_to_building_room(s.설치장소)[0]
        s.동선 = route_by_building.get(building)
        s.출동확정 = True
    db.commit()

    return {"확정_일정수": len(rows), "건물수": len(route_results), "동선": route_results}


@app.patch("/schedules/{schedule_id}")
def update_schedule(schedule_id: int, body: SchedulePatch, db: Session = Depends(get_db)):
    """일정·동선 수동 수정 (변경할 필드만 보내면 됨)."""
    s = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다.")
    for field in ("출동일시", "자산번호", "품명", "규격모델", "금액", "설치장소",
                  "신청부서", "수량", "필요인원수", "투입인원수", "가용명단",
                  "출동확정", "동선"):
        val = getattr(body, field)
        if val is not None:
            setattr(s, field, val)
    db.commit()
    db.refresh(s)
    return _schedule_to_dict(s)


# ==========================================
# 건물 내 수거 동선 최적화
# ==========================================
@app.post("/optimize/route")
async def optimize_route(data: RouteRequest):
    if not data.신청서:
        raise HTTPException(status_code=400, detail="신청서 목록이 비어 있습니다.")

    rows = []
    for idx, item in enumerate(data.신청서, start=1):
        building = item.설치장소.split()[0] if item.설치장소 and item.설치장소.strip() else ""
        if not building:
            raise HTTPException(status_code=400, detail=f"신청서 {idx}번: 설치장소에서 건물명을 파악할 수 없습니다. (입력값: '{item.설치장소}')")
        rows.append({
            "신청번호":   str(idx),
            "품명":       item.품명,
            "설치장소":   item.설치장소,
            "수량":       item.수량,
            "필요인원수": item.필요인원수,
        })

    df              = pd.DataFrame(rows)
    dispatch_people = float(data.투입인원수) if data.투입인원수 is not None else None

    try:
        results = run_route_optimizer(df, None, dispatch_people)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"건물 그래프 파일 없음: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최적화 실행 오류: {e}")

    return JSONResponse(content=results)


# ==========================================
# 제품 조회
# ==========================================
@app.get("/products")
def get_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    products = db.query(Product).offset(skip).limit(limit).all()
    return [
        {
            "id":         p.id,
            "품명":       p.품명,
            "필요인원수": p.필요인원수,
        }
        for p in products
    ]


# ==========================================
# 품목별 필요인원수 조회
# ==========================================
@app.get("/products/workers")
def get_workers(품명: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.품명 == 품명).first()
    if not product:
        raise HTTPException(status_code=404, detail="해당 제품을 찾을 수 없습니다.")
    return {"품명": product.품명, "필요인원수": product.필요인원수}


# ==========================================
# 제품 단건 추가
# ==========================================
@app.post("/products")
def create_product(body: ProductCreate, db: Session = Depends(get_db)):
    product = Product(
        품명=body.품명,
        필요인원수=body.필요인원수,
    )
    db.add(product)
    db.commit()
    db.refresh(product)
    return {"id": product.id, "품명": product.품명, "필요인원수": product.필요인원수}


# ==========================================
# 제품 수정
# ==========================================
@app.patch("/products/{name}")
def update_product(name: str, body: ProductUpdate, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.품명 == name).first()
    if not product:
        raise HTTPException(status_code=404, detail="제품을 찾을 수 없습니다.")
    if body.품명 is not None:
        product.품명 = body.품명
    if body.필요인원수 is not None:
        product.필요인원수 = body.필요인원수
    db.commit()
    db.refresh(product)
    return {"id": product.id, "품명": product.품명, "필요인원수": product.필요인원수}


# ==========================================
# S3 CSV → RDS bulk import
# ==========================================
@app.post("/products/import")
def import_products_from_s3(body: ImportRequest, db: Session = Depends(get_db)):
    # 자격증명은 boto3 기본 체인(env → ~/.aws → EC2 IAM 역할)에 맡긴다.
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    try:
        obj = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=body.s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"S3 파일 읽기 실패: {e}")

    required_cols = {"품명", "필요인원수"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV 필수 컬럼 없음: {required_cols - set(df.columns)}")

    records = [
        Product(품명=row["품명"], 필요인원수=int(row["필요인원수"]))
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    db.commit()
    return {"imported": len(records)}
