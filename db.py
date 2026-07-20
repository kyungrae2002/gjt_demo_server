import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# 사용자·비밀번호는 특수문자(@ : / 등)가 URL을 깨뜨리지 않도록 인코딩한다.
# (Supabase 자동 생성 비밀번호 대응)
_user = quote_plus(os.getenv("DB_USER", ""))
_pw   = quote_plus(os.getenv("DB_PASSWORD", ""))

DATABASE_URL = (
    f"postgresql+psycopg2://{_user}:{_pw}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"sslmode": "require"},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
