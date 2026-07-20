from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, func
from db import Base


class Product(Base):
    """품목·인원수 마스터 (필요인원수의 기준값)."""
    __tablename__ = "products"

    id         = Column(Integer, primary_key=True, index=True)
    품명       = Column(String(255), nullable=False)
    필요인원수 = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Application(Base):
    """신청서 (OCR 파싱 데이터). 상태: 접수 → 일정확정 → 완료."""
    __tablename__ = "applications"

    id         = Column(Integer, primary_key=True, index=True)
    신청번호   = Column(String(64), nullable=False, unique=True, index=True)  # 연결 키
    신청일자   = Column(String(32))
    신청부서   = Column(String(255))
    신청자     = Column(String(255), nullable=True)
    연락처     = Column(String(64), nullable=True)
    원본파일명 = Column(String(255), nullable=True)
    물품목록   = Column(JSON, nullable=False, default=list)  # [{품명,설치장소,수량,필요인원수}]
    상태       = Column(String(16), nullable=False, default="접수", index=True)
    점검완료   = Column(Boolean, nullable=False, default=False)  # true라야 최적화 대상
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Schedule(Base):
    """최적화 결과 (수거 일정 + 동선). applications(1)—(N)schedules, 키=신청번호."""
    __tablename__ = "schedules"

    id              = Column(Integer, primary_key=True, index=True)
    신청번호        = Column(String(64), nullable=False, index=True)  # → applications
    출동일시        = Column(DateTime, index=True)  # "오늘 일정" 필터 기준
    품명            = Column(String(255))
    설치장소        = Column(String(255))
    신청부서        = Column(String(255))
    수량            = Column(Integer, default=1)
    필요인원수      = Column(Integer, default=1)
    투입인원수      = Column(Integer, default=0)
    가용명단        = Column(String(1024), nullable=True)
    optimize_run_id = Column(String(64), index=True)  # 재최적화 배치 단위(교체 시 이 키로 폐기)
    출동확정        = Column(Boolean, nullable=False, default=False)
    동선            = Column(JSON, nullable=True)  # 건물별 방문 순서(model2.py 결과)
    created_at      = Column(DateTime, server_default=func.now())
