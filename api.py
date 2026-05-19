import io
import os
import pandas as pd
import boto3
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from model import run_optimizer
from db import get_db, engine, Base
from models import Product

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://bodies-copied-lit-namely.trycloudflare.com",
        "https://gwanzae.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic 스키마
# ==========================================
class ProductCreate(BaseModel):
    제품이름: str
    자산번호: str
    필요인원수: int

class ProductUpdate(BaseModel):
    제품이름: Optional[str] = None
    자산번호: Optional[str] = None
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
# 최적화
# ==========================================
@app.post("/optimize")
async def optimize(data: List[Dict[str, Any]]):
    rows = []
    for req in data:
        for item in req["물품목록"]:
            rows.append({
                "신청번호":   req["신청번호"],
                "신청일자":   req["신청일자"],
                "신청부서":   req["신청부서"],
                "품명":       item["품명"],
                "설치장소":   item["설치장소"],
                "수량":       item["수량"],
                "필요인원수": item["필요인원수"],
            })
    df       = pd.DataFrame(rows)
    df_avail = pd.read_csv("datas/근로학생시간.csv")
    results  = run_optimizer(df, df_avail)
    return JSONResponse(content=results)


# ==========================================
# 제품 조회
# ==========================================
@app.get("/products")
def get_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    return [
        {
            "id":         p.id,
            "제품이름":   p.제품이름,
            "자산번호":     p.자산번호,
            "필요인원수": p.필요인원수,
        }
        for p in products
    ]


# ==========================================
# 제품 단건 추가
# ==========================================
@app.post("/products")
def create_product(body: ProductCreate, db: Session = Depends(get_db)):
    product = Product(
        제품이름=body.제품이름,
        자산번호=body.자산번호,
        필요인원수=body.필요인원수,
    )
    db.add(product)
    db.commit()
    db.refresh(product)
    return {"id": product.id, "제품이름": product.제품이름, "자산번호": product.자산번호, "필요인원수": product.필요인원수}


# ==========================================
# 제품 수정
# ==========================================
@app.put("/products/{serialnumber}")
def update_product(serialnumber: str, body: ProductUpdate, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.자산번호 == serialnumber).first()
    if not product:
        raise HTTPException(status_code=404, detail="제품을 찾을 수 없습니다.")
    if body.제품이름 is not None:
        product.제품이름 = body.제품이름
    if body.자산번호 is not None:
        product.자산번호 = body.자산번호
    if body.필요인원수 is not None:
        product.필요인원수 = body.필요인원수
    db.commit()
    db.refresh(product)
    return {"id": product.id, "제품이름": product.제품이름, "자산번호": product.자산번호, "필요인원수": product.필요인원수}


# ==========================================
# S3 CSV → RDS bulk import
# ==========================================
@app.post("/products/import")
def import_products_from_s3(body: ImportRequest, db: Session = Depends(get_db)):
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        obj = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=body.s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"S3 파일 읽기 실패: {e}")

    required_cols = {"제품이름", "자산번호", "필요인원수"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV 필수 컬럼 없음: {required_cols - set(df.columns)}")

    records = [
        Product(제품이름=row["제품이름"], 자산번호=row["자산번호"], 필요인원수=int(row["필요인원수"]))
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    db.commit()
    return {"imported": len(records)}
