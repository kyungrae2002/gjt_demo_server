from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float, JSON, func
from db import Base


class Organization(Base):
    """입장 코드로 참여하는 작업 조직(관리자 1명 이상 + 작업자 여러 명)."""

    __tablename__ = "organizations"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(100), nullable=False)
    entry_code = Column(String(6), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, server_default=func.now())


class Product(Base):
    """품목·인원수 마스터 (필요인원수의 기준값)."""
    __tablename__ = "products"

    id         = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    품명       = Column(String(255), nullable=False)
    필요인원수 = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Application(Base):
    """신청서 (OCR 파싱 데이터). 상태: 접수 → 일정확정 → 완료."""
    __tablename__ = "applications"

    id         = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
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
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    신청번호        = Column(String(64), nullable=False, index=True)  # → applications
    출동일시        = Column(DateTime, index=True)  # "오늘 일정" 필터 기준
    자산번호        = Column(String(64), nullable=True)
    품명            = Column(String(255))
    규격모델        = Column(String(255), nullable=True)
    금액            = Column(String(64), nullable=True)
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


class User(Base):
    """로그인 계정. 비밀번호 원문과 재설정 토큰 원문은 저장하지 않는다."""

    __tablename__ = "users"

    id                     = Column(Integer, primary_key=True, index=True)
    organization_id        = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    role                   = Column(String(16), nullable=False, default="worker", index=True)
    username               = Column(String(50), nullable=False, unique=True, index=True)
    email                  = Column(String(255), nullable=False, unique=True, index=True)
    full_name              = Column(String(100), nullable=False)
    password_hash          = Column(String(255), nullable=False)
    is_active              = Column(Boolean, nullable=False, default=True)
    reset_token_hash       = Column(String(64), nullable=True, unique=True, index=True)
    reset_token_expires_at = Column(DateTime, nullable=True)
    created_at             = Column(DateTime, server_default=func.now())
    updated_at             = Column(DateTime, server_default=func.now(), onupdate=func.now())


class WorkSession(Base):
    """근로자별 실제 출동 측정 결과와 작업 후 Borg 응답."""

    __tablename__ = "work_sessions"

    id                  = Column(Integer, primary_key=True, index=True)
    organization_id     = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    client_session_id   = Column(String(64), nullable=False, unique=True, index=True)
    worker_name         = Column(String(100), nullable=False, index=True)
    schedule_ids        = Column(JSON, nullable=False, default=list)
    application_numbers = Column(JSON, nullable=False, default=list)
    started_at          = Column(DateTime, nullable=False, index=True)
    completed_at        = Column(DateTime, nullable=False, index=True)
    total_seconds       = Column(Integer, nullable=False)
    work_seconds        = Column(Integer, nullable=True)
    driving_seconds     = Column(Integer, nullable=False, default=0)
    unknown_seconds     = Column(Integer, nullable=False, default=0)
    gps_sample_count    = Column(Integer, nullable=False, default=0)
    gps_rejected_count  = Column(Integer, nullable=False, default=0)
    tracking_quality    = Column(String(32), nullable=False, default="unavailable")
    borg_cr10           = Column(Integer, nullable=True)
    borg_source         = Column(String(16), nullable=True)
    predicted_borg_cr10 = Column(Float, nullable=True)
    prediction_confidence = Column(String(16), nullable=True)
    prediction_model_version = Column(String(32), nullable=True)
    prediction_validation_mae = Column(Float, nullable=True)
    feature_snapshot    = Column(JSON, nullable=True)
    created_at          = Column(DateTime, server_default=func.now())


class FatigueModel(Base):
    """조직 공통 또는 작업자 개인별 피로 예측 모델 스냅샷.

    실제 작업 기록은 ``work_sessions``에 보존하고, 여기에는 재사용 가능한
    학습 파라미터와 관리자 화면의 초기 참고값만 저장한다.
    """

    __tablename__ = "fatigue_models"

    id                    = Column(Integer, primary_key=True, index=True)
    organization_id       = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id               = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    worker_name           = Column(String(100), nullable=True, index=True)
    scope                 = Column(String(16), nullable=False, default="personal", index=True)
    model_version         = Column(String(32), nullable=False)
    feature_schema        = Column(JSON, nullable=False, default=list)
    parameters            = Column(JSON, nullable=True)
    baseline_borg_cr10    = Column(Float, nullable=True)
    source                = Column(String(32), nullable=False, default="operational")
    training_sample_count = Column(Integer, nullable=False, default=0)
    actual_response_count = Column(Integer, nullable=False, default=0)
    validation_count      = Column(Integer, nullable=False, default=0)
    validation_mae        = Column(Float, nullable=True)
    is_active             = Column(Boolean, nullable=False, default=True, index=True)
    trained_at            = Column(DateTime, nullable=False, server_default=func.now())
    updated_at            = Column(DateTime, server_default=func.now(), onupdate=func.now())


class StaffingDecision(Base):
    """관리자가 추천안을 확인한 뒤 명시적으로 확정한 출동별 인원 구성."""

    __tablename__ = "staffing_decisions"

    id                      = Column(Integer, primary_key=True, index=True)
    organization_id         = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    confirmed_by_user_id    = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    dispatch_time           = Column(DateTime, nullable=False, index=True)
    schedule_ids            = Column(JSON, nullable=False, default=list)
    selected_workers        = Column(JSON, nullable=False, default=list)
    recommendation_snapshot = Column(JSON, nullable=True)
    confirmed_at            = Column(DateTime, nullable=False, server_default=func.now())
