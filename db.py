import os
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


DEMO_MODE = _env_flag("DEMO_MODE")

# 사용자·비밀번호는 특수문자(@ : / 등)가 URL을 깨뜨리지 않도록 인코딩한다.
# (Supabase 자동 생성 비밀번호 대응)
_user = quote_plus(os.getenv("DB_USER", ""))
_pw   = quote_plus(os.getenv("DB_PASSWORD", ""))

if DEMO_MODE:
    # 시연은 PostgreSQL/Supabase에 절대 연결하지 않고 저장소 내부 SQLite 파일만 사용한다.
    _demo_db_path = Path(os.getenv("DEMO_DB_PATH", "test/demo.sqlite3"))
    if not _demo_db_path.is_absolute():
        _demo_db_path = Path(__file__).resolve().parent / _demo_db_path
    _demo_db_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite:///{_demo_db_path}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
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
