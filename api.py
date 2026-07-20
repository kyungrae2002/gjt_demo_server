import io
import os
from datetime import date, datetime
import yaml
import pandas as pd
import boto3
from fastapi import FastAPI, Depends, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from model import run_optimizer
from model2 import run_optimizer as run_route_optimizer
from db import get_db, engine, Base
from models import Product, Application
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

# Origin 검증을 건너뛸 경로 (브라우저 주소창에서 직접 여는 문서·헬스체크)
ORIGIN_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/openapi.yaml"}


@app.middleware("http")
async def verify_origin(request: Request, call_next):
    # 문서·헬스체크는 브라우저에서 직접 열 수 있어야 하므로 검증 제외
    if request.url.path in ORIGIN_EXEMPT_PATHS:
        return await call_next(request)
    # CORS preflight(OPTIONS)는 CORSMiddleware가 처리하도록 통과
    if request.method == "OPTIONS":
        return await call_next(request)
    # 프론트엔드(cross-origin) 요청에는 브라우저가 항상 Origin 헤더를 붙인다
    origin = request.headers.get("origin")
    if origin not in ALLOWED_ORIGINS:
        return JSONResponse(
            status_code=403,
            content={"detail": "허용되지 않은 요청 출처입니다."},
        )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# 신청서 OCR 저장 (NAVER CLOVA OCR)
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


def _enrich_items_with_master(items: list, db: Session) -> list:
    """각 품목의 필요인원수를 products 마스터(품명)로 채운다. 없으면 OCR값→1."""
    enriched = []
    for it in items:
        name = (it.get("품명") or "").strip()
        master = db.query(Product).filter(Product.품명 == name).first()
        ppl = master.필요인원수 if master else int(it.get("필요인원수") or 1)
        enriched.append({
            "품명":       name,
            "설치장소":   (it.get("설치장소") or "").strip(),
            "수량":       int(it.get("수량") or 1),
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
    신청서 이미지(PNG/JPEG/TIFF) 또는 PDF를 업로드하면
    CLOVA OCR로 기본정보·품목 표를 추출해 applications 에 저장한다. (상태=접수, 점검완료=false)

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
        raise HTTPException(status_code=502, detail=f"OCR(CLOVA) 처리 실패: {e}")

    items = parsed["items"]
    if not items:
        raise HTTPException(
            status_code=422,
            detail="신청서에서 품목 표를 인식하지 못했습니다. 표 형태/화질을 확인해 주세요.",
        )

    hdr = parsed["header"]
    # 폼 값 > OCR 헤더 > 기본값 순으로 채운다.
    num = 신청번호 or hdr.get("신청번호") or _gen_application_no()
    day = _normalize_date(신청일자 or hdr.get("신청일자"))
    first_place = items[0]["설치장소"].strip() if items[0].get("설치장소") else ""
    dept = 신청부서 or hdr.get("신청부서") or (first_place.split()[0] if first_place else "창고")
    applicant = 신청자 or hdr.get("신청자")
    contact = 연락처 or hdr.get("연락처")

    if db.query(Application).filter(Application.신청번호 == num).first():
        raise HTTPException(status_code=409, detail=f"이미 존재하는 신청번호입니다: {num}")

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
