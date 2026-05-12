import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import run_optimizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/results")
def get_results():
    df = pd.read_csv("datas/출동결과.csv", encoding="utf-8-sig")
    return JSONResponse(content=df.to_dict(orient="records"))


@app.post("/optimize")
async def optimize(
    request_file: UploadFile = File(..., description="불용신청_생성데이터.csv"),
):
    df       = pd.read_csv(io.BytesIO(await request_file.read()), encoding="utf-8-sig")
    df_avail = pd.read_csv("datas/근로학생시간.csv")
    results  = run_optimizer(df, df_avail)
    return JSONResponse(content=results)
