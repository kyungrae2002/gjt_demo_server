# GJT ML Server

물품 배치 최적화 및 제품 관리 API 서버입니다.

---

## 실행 전 준비사항

### 1. Python 설치 확인
터미널(PowerShell)을 열고 아래 명령어를 입력하세요.
```
python --version
```
`Python 3.10` 이상이 출력되면 됩니다.

---

### 2. 가상환경 생성 및 활성화
프로젝트 루트 폴더(`gjt_ml_server`)에서 실행합니다.

```
python -m venv venv
venv\Scripts\activate
```

활성화되면 터미널 앞에 `(venv)` 가 붙습니다.

---

### 3. 패키지 설치
```
pip install -r gjt_demo_server/requirements.txt
```

---

### 4. 환경변수 파일 설정
`gjt_demo_server` 폴더 안에 `.env` 파일을 만들고 아래 내용을 채웁니다.

```
DB_HOST=<RDS 엔드포인트>
DB_PORT=5432
DB_NAME=postgres
DB_USER=<DB 사용자 이름>
DB_PASSWORD=<DB 비밀번호>

AWS_ACCESS_KEY_ID=<AWS 액세스 키>
AWS_SECRET_ACCESS_KEY=<AWS 시크릿 키>
AWS_REGION=us-east-1
S3_BUCKET=<S3 버킷 이름>
```

> `.env` 파일은 Git에 올라가지 않습니다. 팀원에게 별도로 전달받으세요.

---

## 서버 실행

`gjt_demo_server` 폴더로 이동 후 실행합니다.

```
cd gjt_demo_server
uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

터미널에 아래 메시지가 뜨면 정상입니다.
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

브라우저에서 `http://localhost:8080/docs` 접속 시 API 명세서를 확인할 수 있습니다.

---

## 외부 접속 (Cloudflare Tunnel)

외부(프론트엔드 등)에서 접근하려면 Cloudflare Tunnel을 사용합니다.

### 설치
```
winget install cloudflare.cloudflared
```

### 실행 (서버와 별도 터미널에서)
```
cloudflared tunnel --url http://localhost:8080
```

실행하면 아래와 같이 임시 URL이 생성됩니다.
```
https://xxxx-xxxx-xxxx.trycloudflare.com
```

> 이 URL은 실행할 때마다 바뀝니다. 바뀐 URL을 `api.py`의 `allow_origins`와 프론트엔드에 업데이트해야 합니다.

### 종료
터미널에서 `Ctrl + C`

---

## API 명세

### 공통
- Base URL: `http://localhost:8080` (로컬) 또는 Cloudflare Tunnel URL
- 모든 요청/응답은 `Content-Type: application/json`

---

### GET /health
서버 상태 확인

**Response**
```json
{ "status": "ok" }
```

---

### POST /optimize
물품 배치 최적화 실행

**Request Body**
```json
[
  {
    "신청번호": "2024-001",
    "신청일자": "2024-01-15",
    "신청부서": "행정팀",
    "물품목록": [
      {
        "품명": "책상",
        "설치장소": "본관 101호",
        "수량": 3,
        "필요인원수": 2
      }
    ]
  }
]
```

**Response**
```json
[
  {
    "출동일시": "2024-01-15 09:00",
    "신청서번호": "2024-001",
    "신청부서": "행정팀",
    "설치장소": "본관 101호",
    "품명": "책상",
    "수량": 3,
    "가용명단": "홍길동, 김철수",
    "투입인원수": 2
  }
]
```

---

### GET /products
전체 제품 목록 조회

**Query Parameters** (선택)
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| skip | 0 | 건너뛸 개수 |
| limit | 100 | 가져올 최대 개수 |

**예시**
```
GET /products?skip=0&limit=20
```

**Response**
```json
[
  {
    "id": 1,
    "품명": "책상",
    "필요인원수": 2
  }
]
```

---

### GET /products/workers
품목명으로 필요인원수 조회

**Query Parameters**
| 파라미터 | 필수 | 설명 |
|----------|------|------|
| 품명 | O | 조회할 품목명 |

**예시**
```
GET /products/workers?품명=책상
```

**Response**
```json
{
  "품명": "책상",
  "필요인원수": 2
}
```

**Error (404)**
```json
{ "detail": "해당 제품을 찾을 수 없습니다." }
```

---

### POST /products
제품 단건 추가

**Request Body**
```json
{
  "품명": "책상",
  "필요인원수": 2
}
```

**Response**
```json
{
  "id": 1,
  "품명": "책상",
  "필요인원수": 2
}
```

---

### PATCH /products/{품명}
제품 정보 수정 (변경할 필드만 보내면 됩니다)

**URL 예시**
```
PATCH /products/책상
```

**Request Body** (변경할 필드만)
```json
{
  "필요인원수": 3
}
```

**Response**
```json
{
  "id": 1,
  "품명": "책상",
  "필요인원수": 3
}
```

**Error (404)**
```json
{ "detail": "제품을 찾을 수 없습니다." }
```

---

### POST /products/import
S3에 업로드된 CSV 파일을 DB에 일괄 저장

**Request Body**
```json
{
  "s3_key": "products_master.csv"
}
```

CSV 파일은 반드시 `품명`, `필요인원수` 컬럼을 포함해야 합니다.

**Response**
```json
{ "imported": 150 }
```

---

## 에러 코드

| 상태코드 | 의미 |
|----------|------|
| 200 | 성공 |
| 404 | 해당 리소스를 찾을 수 없음 |
| 422 | 요청 형식이 잘못됨 (필드명, 타입 오류 등) |
| 400 | S3 파일 오류 |

---

## 파일 구조

```
gjt_ml_server/
├── venv/                  # 가상환경 (Git 제외)
└── gjt_demo_server/
    ├── api.py             # API 엔드포인트
    ├── model.py           # LP 최적화 모델
    ├── models.py          # DB ORM 모델
    ├── db.py              # DB 연결 설정
    ├── .env               # 환경변수 (Git 제외)
    ├── requirements.txt   # 패키지 목록
    └── datas/
        └── 근로학생시간.csv  # 최적화에 사용되는 고정 데이터
```
