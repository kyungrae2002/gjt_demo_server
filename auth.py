"""계정 생성, 로그인, 계정 복구 API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import get_db
from fatigue_store import seed_initial_fatigue_models
from models import FatigueModel, Organization, User, WorkSession


router = APIRouter(prefix="/auth", tags=["인증"])
_bearer = HTTPBearer(auto_error=False)

ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "60"))
PASSWORD_RESET_MINUTES = int(os.getenv("PASSWORD_RESET_MINUTES", "15"))
RETURN_RESET_TOKEN = os.getenv("PASSWORD_RESET_RETURN_TOKEN", "").strip().lower() in {
    "1", "true", "t", "yes", "y", "on",
}

_configured_secret = os.getenv("AUTH_SECRET_KEY") or os.getenv("API_KEY")
if _configured_secret:
    _AUTH_SECRET = _configured_secret.encode("utf-8")
else:
    # 개발 환경의 무설정 실행만 지원한다. 프로세스를 재시작하면 기존 토큰은 만료된다.
    _AUTH_SECRET = secrets.token_bytes(48)
    print("[경고] AUTH_SECRET_KEY가 없어 임시 인증 키를 사용합니다.")

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{4,50}$")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_ENTRY_CODE_RE = re.compile(r"^[A-Z0-9]{6}$")
_ENTRY_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _utcnow() -> datetime:
    """DB 드라이버 간 비교가 일관되도록 naive UTC를 사용한다."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_username(value: str) -> str:
    value = value.strip().lower()
    if not _USERNAME_RE.fullmatch(value):
        raise ValueError("아이디는 영문, 숫자, 마침표, 밑줄, 하이픈으로 4~50자여야 합니다.")
    return value


def _normalize_email(value: str) -> str:
    value = value.strip().lower()
    if len(value) > 255 or not _EMAIL_RE.fullmatch(value):
        raise ValueError("올바른 이메일 주소를 입력해 주세요.")
    return value


def _validate_password(value: str) -> str:
    if len(value) < 8 or len(value) > 128:
        raise ValueError("비밀번호는 8~128자여야 합니다.")
    if not any(ch.isalpha() for ch in value) or not any(ch.isdigit() for ch in value):
        raise ValueError("비밀번호에는 문자와 숫자가 각각 하나 이상 필요합니다.")
    return value


class AccountCreate(BaseModel):
    username: str
    password: str
    full_name: str = Field(min_length=1, max_length=100)
    email: str
    organization_mode: Literal["create", "join"] = "create"
    organization_name: Optional[str] = Field(default=None, max_length=100)
    entry_code: Optional[str] = None

    _username = field_validator("username")(_normalize_username)
    _password = field_validator("password")(_validate_password)
    _email = field_validator("email")(_normalize_email)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("이름을 입력해 주세요.")
        return value

    @field_validator("organization_name")
    @classmethod
    def validate_organization_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("entry_code")
    @classmethod
    def validate_entry_code(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().upper()
        if not _ENTRY_CODE_RE.fullmatch(value):
            raise ValueError("입장 코드는 영문 대문자와 숫자로 구성된 6자리여야 합니다.")
        return value


class LoginRequest(BaseModel):
    username: str
    password: str

    _username = field_validator("username")(_normalize_username)


class ForgotPasswordRequest(BaseModel):
    email: str

    _email = field_validator("email")(_normalize_email)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=32, max_length=512)
    new_password: str

    _password = field_validator("new_password")(_validate_password)


class FindIdRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=100)
    email: str

    _email = field_validator("email")(_normalize_email)

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        return value.strip()


class AccountResponse(BaseModel):
    id: int
    username: str
    full_name: str
    email: str
    organization_id: int
    organization_name: str
    role: Literal["admin", "worker"]
    organization_entry_code: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AccountResponse


class MessageResponse(BaseModel):
    message: str


class ForgotPasswordResponse(MessageResponse):
    reset_token: Optional[str] = None


class FindIdResponse(BaseModel):
    username: str


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${_b64encode(salt)}${_b64encode(derived)}"


def _decode_b64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode_b64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_decode_b64(expected)),
        )
        return hmac.compare_digest(actual, _decode_b64(expected))
    except (TypeError, ValueError):
        return False


# 없는 계정의 로그인도 비밀번호 검증 비용을 지불해 계정 존재 여부 추측을 어렵게 한다.
_DUMMY_PASSWORD_HASH = _hash_password(secrets.token_urlsafe(24))


def _create_access_token(user: User) -> str:
    now = int(_utcnow().timestamp())
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "organization_id": user.organization_id,
        "role": user.role,
        "iat": now,
        "exp": now + ACCESS_TOKEN_MINUTES * 60,
        "type": "access",
    }
    encoded_payload = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = _b64encode(
        hmac.new(_AUTH_SECRET, encoded_payload.encode("ascii"), hashlib.sha256).digest()
    )
    return f"v1.{encoded_payload}.{signature}"


def _decode_access_token(token: str) -> dict:
    """서명·만료·용도를 검증하고 액세스 토큰 payload를 반환한다."""
    try:
        version, encoded_payload, encoded_signature = token.split(".", 2)
        if version != "v1":
            raise ValueError
        expected_signature = _b64encode(
            hmac.new(_AUTH_SECRET, encoded_payload.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(encoded_signature, expected_signature):
            raise ValueError
        payload = json.loads(_decode_b64(encoded_payload).decode("utf-8"))
        if payload.get("type") != "access" or int(payload.get("exp", 0)) <= int(_utcnow().timestamp()):
            raise ValueError
        int(payload["sub"])
        return payload
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요하거나 인증이 만료되었습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_access_token(credentials.credentials)
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active or user.username != payload.get("username"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 사용자입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _mask_username(username: str) -> str:
    if len(username) <= 2:
        return username[0] + "*" * (len(username) - 1)
    visible = max(1, min(3, len(username) // 2))
    return username[:visible] + "*" * (len(username) - visible)


def _smtp_is_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def _send_reset_email(user: User, token: str) -> None:
    reset_base_url = os.getenv("PASSWORD_RESET_URL", "http://localhost:3000/reset-password")
    separator = "&" if "?" in reset_base_url else "?"
    reset_url = f"{reset_base_url}{separator}token={quote(token)}"

    message = EmailMessage()
    message["Subject"] = "비밀번호 재설정 안내"
    message["From"] = os.environ["SMTP_FROM"]
    message["To"] = user.email
    message.set_content(
        f"{user.full_name}님, 아래 주소에서 {PASSWORD_RESET_MINUTES}분 이내에 비밀번호를 재설정해 주세요.\n\n"
        f"{reset_url}\n\n본인이 요청하지 않았다면 이 메일을 무시해 주세요."
    )

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    timeout = float(os.getenv("SMTP_TIMEOUT_SECONDS", "10"))
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")

    with smtplib.SMTP(host, port, timeout=timeout) as smtp:
        if use_tls:
            smtp.starttls()
        if username:
            smtp.login(username, password or "")
        smtp.send_message(message)


def _account_response(user: User, db: Session) -> AccountResponse:
    organization = db.get(Organization, user.organization_id) if user.organization_id else None
    if organization is None:
        raise HTTPException(status_code=409, detail="계정에 연결된 조직이 없습니다.")
    return AccountResponse(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        organization_id=organization.id,
        organization_name=organization.name,
        role=user.role,
        organization_entry_code=organization.entry_code if user.role == "admin" else None,
    )


def _generate_entry_code(db: Session) -> str:
    for _ in range(100):
        code = "".join(secrets.choice(_ENTRY_CODE_ALPHABET) for _ in range(6))
        if not db.query(Organization.id).filter(Organization.entry_code == code).first():
            return code
    raise HTTPException(status_code=503, detail="조직 입장 코드를 생성하지 못했습니다.")


@router.post("/signup", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
def signup(body: AccountCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="이미 사용 중인 아이디입니다.")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다.")

    if body.organization_mode == "join":
        if not body.entry_code:
            raise HTTPException(status_code=422, detail="조직 입장 코드를 입력해 주세요.")
        organization = db.query(Organization).filter(Organization.entry_code == body.entry_code).first()
        if not organization:
            raise HTTPException(status_code=404, detail="입장 코드와 일치하는 조직을 찾을 수 없습니다.")
        # 기존 시연 조직처럼 아직 구성원이 없는 조직은 첫 참여자가 관리자가 된다.
        has_member = db.query(User.id).filter(User.organization_id == organization.id).first() is not None
        role = "worker" if has_member else "admin"
    else:
        organization = Organization(
            name=body.organization_name or f"{body.full_name} 조직",
            entry_code=_generate_entry_code(db),
        )
        db.add(organization)
        db.flush()
        role = "admin"

    user = User(
        username=body.username,
        email=body.email,
        full_name=body.full_name,
        password_hash=_hash_password(body.password),
        organization_id=organization.id,
        role=role,
    )
    db.add(user)
    try:
        db.flush()
        if role == "worker":
            # 가입 전 이름으로 쌓아 둔 테스트 초기값·레거시 기록을 실제 계정에 연결한다.
            db.query(WorkSession).filter(
                WorkSession.organization_id == organization.id,
                WorkSession.user_id.is_(None),
                WorkSession.worker_name == user.full_name,
            ).update({WorkSession.user_id: user.id}, synchronize_session=False)
            db.query(FatigueModel).filter(
                FatigueModel.organization_id == organization.id,
                FatigueModel.user_id.is_(None),
                FatigueModel.worker_name == user.full_name,
                FatigueModel.scope == "personal",
            ).update({FatigueModel.user_id: user.id}, synchronize_session=False)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="아이디 또는 이메일이 이미 사용 중입니다.")
    db.refresh(user)
    if body.organization_mode == "create":
        # 새 조직도 실제 개인 응답이 쌓이기 전 공통 초기 모델로 예측할 수 있다.
        seed_initial_fatigue_models(db, organization.id, include_worker_baselines=False)
    return _account_response(user, db)


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    encoded_hash = user.password_hash if user else _DUMMY_PASSWORD_HASH
    password_matches = _verify_password(body.password, encoded_hash)
    if not user or not password_matches or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return LoginResponse(
        access_token=_create_access_token(user),
        expires_in=ACCESS_TOKEN_MINUTES * 60,
        user=_account_response(user, db),
    )


@router.get("/me", response_model=AccountResponse)
def me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _account_response(current_user, db)


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return current_user


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
    response_model_exclude_none=True,
)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    if not _smtp_is_configured() and not RETURN_RESET_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="비밀번호 재설정 메일 발송 설정이 필요합니다.",
        )

    user = db.query(User).filter(User.email == body.email, User.is_active.is_(True)).first()
    generic_message = "가입된 계정이 있으면 비밀번호 재설정 안내를 발송했습니다."
    if not user:
        return ForgotPasswordResponse(message=generic_message)

    token = secrets.token_urlsafe(32)
    user.reset_token_hash = _hash_reset_token(token)
    user.reset_token_expires_at = _utcnow() + timedelta(minutes=PASSWORD_RESET_MINUTES)
    db.commit()

    if _smtp_is_configured():
        try:
            _send_reset_email(user, token)
        except (OSError, smtplib.SMTPException) as exc:
            user.reset_token_hash = None
            user.reset_token_expires_at = None
            db.commit()
            raise HTTPException(status_code=503, detail="비밀번호 재설정 메일을 발송하지 못했습니다.") from exc

    return ForgotPasswordResponse(
        message=generic_message,
        reset_token=token if RETURN_RESET_TOKEN else None,
    )


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = _hash_reset_token(body.token)
    user = db.query(User).filter(User.reset_token_hash == token_hash).first()
    if (
        not user
        or not user.reset_token_expires_at
        or user.reset_token_expires_at < _utcnow()
        or not user.is_active
    ):
        raise HTTPException(status_code=400, detail="재설정 토큰이 유효하지 않거나 만료되었습니다.")

    user.password_hash = _hash_password(body.new_password)
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    db.commit()
    return MessageResponse(message="비밀번호가 변경되었습니다.")


@router.post("/find-id", response_model=FindIdResponse)
def find_id(body: FindIdRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.full_name == body.full_name,
        User.email == body.email,
        User.is_active.is_(True),
    ).first()
    if not user:
        raise HTTPException(status_code=404, detail="일치하는 계정을 찾을 수 없습니다.")
    return FindIdResponse(username=_mask_username(user.username))
