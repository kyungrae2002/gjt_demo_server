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

### 인증 API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/auth/signup` | 조직 생성(관리자) 또는 입장 코드 참여(작업자) 계정 생성 |
| POST | `/auth/login` | 로그인 및 Bearer 액세스 토큰 발급 |
| POST | `/auth/forgot-password` | 이메일로 일회용 비밀번호 재설정 링크 발송 |
| POST | `/auth/reset-password` | 일회용 토큰으로 새 비밀번호 설정 |
| POST | `/auth/find-id` | 이름과 이메일로 마스킹된 아이디 찾기 |

관리자가 조직을 만들며 계정을 생성하는 예시:

```json
{
  "username": "gildong",
  "password": "password123",
  "full_name": "홍길동",
  "email": "gildong@example.com",
  "organization_mode": "create",
  "organization_name": "관재팀"
}
```

작업자가 관리자가 공유한 6자리 입장 코드로 참여하는 예시:

```json
{
  "username": "worker01",
  "password": "password123",
  "full_name": "김작업",
  "email": "worker@example.com",
  "organization_mode": "join",
  "entry_code": "ABC234"
}
```

관리자는 로그인 응답에서 입장 코드를 확인할 수 있고, 작업자는 같은 조직의 신청서·일정·
작업 기록만 조회합니다. `/workers/status/today`와 인원 추천·확정 API는 관리자만 사용할 수 있습니다.

### 개인화 피로도 API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/fatigue/dataset-template` | 사전 학습 데이터셋의 열·단위·별칭 규칙 조회(관리자) |
| POST | `/fatigue/train-dataset` | CSV·JSON·XLSX를 업로드해 공통·개인 모델 즉시 학습(관리자) |
| POST | `/fatigue/predict` | 현재 작업 특성으로 개인별 작업 후 Borg와 설문 필요 여부 예측 |
| POST | `/work-sessions` | 실제 Borg와 사전 예측값을 분리해 개인 작업 기록 저장 |
| GET | `/workers/status/today` | 관리자 조직의 실제/예측 Borg, 회복 반영 당일부하 조회 |

개인 모델은 실제 Borg 8회 이상, 사전 예측 검증 5회 이상, 최근 5회 MAE 1.0 이하가 모두
유지되면 설문을 자동 생략합니다. 그전에는 실제 응답을 받아 모델을 계속 검증합니다.

#### 사전 데이터셋으로 즉시 학습

관리자 Bearer 토큰으로 `POST /fatigue/train-dataset`의 multipart `file` 필드에 CSV,
JSON 또는 XLSX 파일 하나를 업로드합니다. 샘플 CSV는
[`datas/fatigue_training_template.csv`](datas/fatigue_training_template.csv)에 있습니다.

현재 학습 열과 단위는 다음과 같습니다.

| 표준 열 | 한국어 열 예시 | 필수 | 의미 |
|---------|----------------|------|------|
| `worker_name` | 작업자, 작업자명, 이름 | 선택 | 개인 모델을 구분할 이름 |
| `work_minutes` | 작업시간, 실작업시간 | 필수 | 실제 작업 시간(분) |
| `total_minutes` | 총시간, 경과시간 | 선택 | 전체 시간(분), 없으면 세 구간의 합 |
| `driving_minutes` | 차량이동시간, 운전시간 | 선택 | 차량 이동 시간(분), 기본 0 |
| `unknown_minutes` | 미분류시간, 판단불가시간 | 선택 | 구간 미분류 시간(분), 기본 0 |
| `item_count` | 물품개수, 일정개수, 개수 | 필수 | 작업 대상 개수 |
| `labor_load` | 노동량, 필요인원합계 | 필수 | 기존 노동량 값 |
| `team_size` | 투입인원수, 작업인원수 | 필수 | 실제 투입 인원 |
| `borg_cr10` | 예상피로도, 피로도, Borg10 | 필수 | 학습 목표값(0~10) |

`started_at`/`completed_at`(또는 `출동일시`/`측정시점`)이 있으면 다음 두 값을 추가
학습 특성으로 사용합니다.

- `cumulative_fatigue`: 같은 날 완료된 이전 출동의 Borg를 현재 출동 시작 시각까지
  개인 반감기로 감쇠해 합산합니다. 첫 출동과 날짜가 바뀐 첫 출동은 0입니다.
- `recovery_minutes`: 같은 날 직전 출동 완료부터 현재 출동 시작까지의 시간입니다.
  첫 출동은 0입니다.

누적피로도 계산식은 `Σ(이전 출동 Borg × 0.5^(경과분/개인 반감기))`입니다.
기존 클라이언트 호환을 위해 `recovered_daily_load`도 같은 누적피로도 값으로 반환합니다.

| 작업자 | 회복 반감기(분) | 회복 속도 순서 |
|--------|-----------------|----------------|
| 김경언 | 60 | 1 (가장 빠름) |
| 최현 | 90 | 2 |
| 강경래 | 120 | 3 |
| 정하람 | 180 | 4 |
| 박우민 | 360 | 5 (현저히 느림) |

표의 5명은 위 고정 운영계수를 사용합니다. 그 외 작업자는 동일 날짜 연속 출동 데이터가
충분할 때 제한적으로 반감기를 보정하며, 부족하면 120분을 사용합니다.

`datas/이지픽업_CR10_모델입력_현재최종.xlsx`는 `01_모델입력` 시트를 자동 인식합니다.
참여 작업자의 `작업후` 행만 사용하고, 묶음 신청의 작업시간을 중복 집계하지 않도록
`근로학생 + 최종출동ID` 단위로 합칩니다. 묶음 안의 Borg는 안전 측면에서 최댓값을
출동 결과로 사용합니다. `누적피로도`와 `휴식시간(분)` 열이 채워진 파일을 업로드하면
서버가 같은 규칙으로 독립 계산한 값과 대조하고, 불일치하면 학습을 중단합니다. 두 열이
비어 있어도 서버가 출동 시각과 Borg로 계산합니다.

전체 데이터가 3행 이상이면 조직 공통 모델을 만들고, 동일한 `worker_name` 데이터가
3행 이상이면 해당 개인 모델도 함께 만듭니다. 이름이 가입된 작업자와 같으면 `user_id`에
즉시 연결하고, 아직 가입하지 않았다면 이름 기반으로 저장합니다. 인식하지 못한 추가 열은
무시되며 응답의 `ignored_columns`에 표시됩니다.

업로드한 예상 피로도는 초기 학습값이므로 실제 Borg 응답 횟수에는 포함되지 않습니다.
따라서 업로드 직후 예측은 가능하지만 설문 자동 생략은 실제 작업 데이터로 기존 검증 기준을
충족한 후에만 이루어집니다. 같은 조직에서 다시 업로드하면 조직 공통 모델과 파일에 포함된
작업자의 활성 개인 모델이 새 학습 결과로 교체됩니다.

5명 계정과 위 엑셀의 개인 모델을 한 트랜잭션에서 연결하려면 대상 DB 설정 후 다음을
실행합니다. 조직이 여러 개면 `--organization-id` 또는 `--entry-code`를 지정해야 합니다.

```bash
venv/bin/python -m scripts.provision_personalized_workers --organization-id 1
```

기본 계정은 `choihyun`, `kimkyungeon`, `kangkyungrae`, `parkwumin`, `jeongharam`이고
초기 비밀번호는 실행 시 안전하게 생성되어 한 번 출력됩니다. 운영 이메일·아이디·비밀번호를
지정하려면 JSON 배열 파일을 만들어 `--account-config`로 전달합니다. 같은 조직에 동일한
이름의 작업자 계정이 하나 있으면 기존 계정을 재사용하고, 둘 이상이면 임의 선택하지 않고
중단합니다.

로그인 요청 예시:

```json
{
  "username": "gildong",
  "password": "password123"
}
```

비밀번호 재설정 메일 발송에는 `.env.example`의 SMTP 설정이 필요합니다. 로컬 개발에서는
`PASSWORD_RESET_RETURN_TOKEN=true`로 설정하면 메일 대신 응답의 `reset_token`을
`POST /auth/reset-password`에 전달해 흐름을 확인할 수 있습니다. 운영에서는 이 옵션을
활성화하지 마세요.

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
