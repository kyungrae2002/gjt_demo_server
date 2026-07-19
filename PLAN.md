# 수거 일정 워크플로우 설계 (기획 진행 문서)

> 상태: **설계 확정 (Q1~Q7 결정 완료)**. 다음 단계 = 구현.

---

## 1. 목표 (사용자 요구사항)

1. OCR로 뽑은 신청서 데이터를 **저장**한다. (재최적화 시 원본이 필요)
2. **오늘의 수거 일정(출동)**을 조회할 수 있어야 한다. (시간대별 렌더링은 프론트)
3. **신청서 목록**을 처리해야 할 순서대로(= 배정된 출동일시 순) 조회한다.
4. 목록에서 **[최적화] 버튼** → 목록 안에서 **다시 최적화**.
5. 최적화 후 **신청서 목록 재렌더링**.
6. **최적화 전 점검 단계**: 인원수뿐 아니라 기본정보(신청부서·신청번호·**신청자·연락처**·신청일)까지 확인·수정.
7. **품목별 인원수 점검**: 품목을 나열하고 각 품목의 필요인원수를 한 페이지에서 확인·수정 후 확정.

### 재최적화가 필요한 진짜 이유 (Q1 확정)
> 아직 **출동(수거)하지 않은** 채 쌓여 있던 출동 목록에, **새 신청서**가 들어오면 새 출동이 추가되며 **기존 출동과 시간대가 충돌**한다.
> → 이때 **기존(미출동) 일정 + 신규 신청서를 합쳐서 전체를 새로 optimize** 해야 충돌이 풀린다.
> → 즉 재최적화 입력 = **"완료(출동)되지 않은 모든 신청서"**. 기존 schedules는 폐기 후 전체 재계산.

---

## 2. 결정 사항 (확정)

| # | 항목 | 결정 |
|---|------|------|
| Q1 | 재최적화 범위 | **미완료(출동 전) 신청서 전체를 합쳐 전체 재계산** |
| Q2 | 처리 상태 | **3단계: 접수 → 일정확정 → 수거완료** |
| Q3 | 물품목록 저장 | **applications 테이블의 JSON 컬럼** |
| Q4 | 실내 동선(model2.py) | **포함**. 단, **출동 확정 후** 동선 계산·저장. 수정 시 DB 업데이트 |
| Q5 | 재최적화 시 확정분 처리 | **고정 없음 — 미완료 전체 재계산** (출동확정분도 다시 흔듦) |
| Q6 | 출동일시 기준일 | **실행일(오늘) 기준** — 과거 슬롯 미발생 |
| Q7 | 기존 `/ocr/optimize` | **폐기** (`/ocr/applications` + `/optimize/run` 2단계로 대체) |

---

## 3. 전체 흐름 (확정 반영)

```
[신청서 이미지 업로드]  POST /ocr/applications
        ▼
   Textract OCR 파싱  →  applications 저장  [상태: 접수, 점검완료=false]
        │
        │  ← [점검]  GET /applications/{신청번호}  (기본정보 + 품목별 인원수 조회)
        │            PATCH /applications/{신청번호}  (기본정보·품목별 인원수 수정 → 점검완료=true)
        │            ※ 최적화는 점검완료=true 인 신청서만 대상
        │
        │  ← [최적화]  POST /optimize/run
        │     입력 = 완료(출동) 아닌 모든 신청서 (접수 + 일정확정 + 출동확정)  ※확정분도 재계산(Q5)
        ▼
   run_optimizer(미완료 신청서 전체, 근로학생시간표)
        │   (기존 schedules 폐기 → 전체 재계산 → 충돌 해소)
        ▼
   schedules 저장 / 갱신  →  applications [상태: 일정확정]
        │
        ├─ GET /applications        처리순서(출동일시 순) 목록  ─┐ 최적화 후 재렌더링
        ├─ GET /schedules/today     오늘 출동 후보              ─┘
        │
        │  ← [오늘 출동 확정]  POST /dispatch/confirm
        ▼
   model2.py 동선 계산(건물별)  →  schedules.동선 저장  [출동확정]
        │   (수정 필요 시  PATCH /schedules/{id}  로 일정·동선 업데이트)
        │
        │  ← [수거 완료]  PATCH /applications/{신청번호}/complete
        ▼
   applications [상태: 완료]  →  이후 재최적화 대상에서 제외
```

- 최적화 JSON에 OCR 필드를 "붙이는" 것은 **메모리 병합**(신청번호 키)이며 DB JOIN 아님. 저장은 재조회·재최적화용.

---

## 4. 데이터 모델 (테이블 3개 / 인스턴스 1개)

### ① `products` — 품목·인원수 마스터 (기존, 변경 없음)
id · 품명 · 필요인원수 · created_at · updated_at

### ② `applications` — 신청서 (OCR 파싱 데이터)
| 컬럼 | 타입 | 비고 |
|------|------|------|
| id | int PK | |
| 신청번호 | str, unique index | 연결 키 |
| 신청일자 | str/date | 기본정보 |
| 신청부서 | str | 기본정보 |
| 신청자 | str, nullable | 기본정보 (신규) |
| 연락처 | str, nullable | 기본정보 (신규) |
| 원본파일명 | str, nullable | |
| 물품목록 | JSON | `[{품명,설치장소,수량,필요인원수}]` (Q3) |
| 상태 | str enum | `접수` / `일정확정` / `완료` (Q2) |
| 점검완료 | bool, default false | 점검 게이트 — true라야 최적화 대상 |
| created_at / updated_at | datetime | |

### ③ `schedules` — 최적화 결과 (수거 일정 + 동선)
| 컬럼 | 타입 | 비고 |
|------|------|------|
| id | int PK | |
| 신청번호 | str index (→ applications) | 연결 키 |
| 출동일시 | datetime | "오늘 일정" 필터 기준 |
| 품명 / 설치장소 / 신청부서 | str | |
| 수량 / 필요인원수 / 투입인원수 | int | |
| 가용명단 | str | |
| optimize_run_id | str | 재최적화 배치 단위(교체 시 이 키로 폐기) |
| 출동확정 | bool, default false | 오늘 출동 확정 여부(Q4) |
| 동선 | JSON, nullable | 건물별 방문 순서(model2.py 결과) — 확정 후 채움 |
| created_at | datetime | |

관계: `applications(1) — (N) schedules`, 키 = **신청번호**.
"신청서 목록 순서" = applications를 schedules.출동일시 기준 정렬.

---

## 5. 상태 전이

```
접수 ──[점검 PATCH]──► 접수(점검완료=true) ──[/optimize/run]──► 일정확정 ──[.../complete]──► 완료
```
- **점검 게이트**: `점검완료=true` 라야 최적화 대상. (상태는 3단계 유지, 점검은 boolean 플래그)
- **재최적화 입력** = `완료`가 아니고 `점검완료=true` 인 신청서 전체(일정확정·출동확정 포함). 출동확정분도 다시 계산(Q5).
- `완료` 처리된 신청서만 목록·재최적화에서 제외.

---

## 6. API 엔드포인트 (제안)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/ocr/applications` | 업로드 → OCR → applications 저장(접수, 점검완료=false) |
| GET | `/applications/{신청번호}` | 점검용 상세: 기본정보 + 품목별 인원수 |
| PATCH | `/applications/{신청번호}` | 기본정보(신청부서/신청자/연락처/신청일)·품목별 인원수 수정, 점검완료 처리. **인원수 수정 시 `products` 마스터도 upsert(일반 적용)** |
| POST | `/optimize/run` | **점검완료** 미완료 신청서 전체 재최적화 → schedules 갱신, 상태=일정확정 |
| GET | `/applications` | 신청서 목록(출동일시 순, 상태 필터) |
| GET | `/schedules/today` | 오늘 출동 일정 |
| POST | `/dispatch/confirm` | 오늘 출동 확정 → model2.py 동선 계산·저장, 출동확정=true |
| PATCH | `/schedules/{id}` | 일정·동선 수동 수정 |
| PATCH | `/applications/{신청번호}/complete` | 수거 완료(상태=완료) |

> 기존 `/ocr/optimize`(OCR+최적화 한 번에)는 **폐기**하고 `/ocr/applications` + `/optimize/run` 2단계로 대체(Q7).

---

## 7. 반드시 손봐야 할 코드

1. **`model.py` 랜덤 20건 추출 제거** ([model.py:26-31](model.py#L26))
   - 재최적화는 미완료 신청서 **전체를 결정적으로** 처리해야 함. `random` 선택 삭제.
2. **최적화 출력에 `신청번호`·`신청일자` 추가** ([model.py:322](model.py#L322))
   - `신청번호=g`, `신청일자=G_data[g]['recv']` 2줄 추가 → 연결 키/저장.
3. **`출동일시` 기준일을 실행일(오늘)로 변경** ([model.py:45](model.py#L45) `ref_date = 접수일자.min()`)
   - `ref_date`를 실행일(오늘) 기준으로 잡아 과거 슬롯이 안 생기게 함(Q6).
4. **model2.py 연동**: 현재 `schedule.json` 파일 입력 → DB(오늘 schedules) 입력으로 연결 필요.
5. **`/ocr/optimize` 폐기**: [api.py](api.py)의 기존 통합 엔드포인트 제거(Q7).
6. **`ocr.py` 헤더 필드 확장** ([ocr.py:33](ocr.py#L33) `HEADER_SYNONYMS`): `신청자`, `연락처` 동의어 추가.
7. **품목별 인원수 = products 마스터로 일반 관리**(확정):
   - 점검 화면 초기값 = `products` 마스터에서 `품명`으로 조회(없으면 OCR값→1).
   - 점검에서 인원수를 수정하면 **`products` 마스터에 반영(upsert by 품명)**.
   - **적용 범위 = 앞으로만**: 갱신된 마스터값은 **이후 조회·신규 신청서부터** 적용. 이미 저장된 다른 신청서 스냅샷은 **소급 변경하지 않음**.
   - 신청서(`applications.물품목록`)에는 조회/최적화 시점 값이 스냅샷으로 남지만, 인원수의 기준(source of truth)은 마스터.
8. **신청자·연락처 OCR**(확정): Textract FORMS로 추출 **시도**, 못 잡으면 점검 화면에서 수동 보정.

---

## 8. 남은 결정

없음 — Q1~Q7 모두 확정. (§2, §7 참고)

---

## 9. 구현 순서 (제안)

1. 모델 추가: `applications`(신청자·연락처·점검완료 포함), `schedules` (+ `model.py` 출력 필드 2줄)
2. `POST /ocr/applications` (OCR 저장) + `ocr.py`에 신청자·연락처 동의어 추가
3. **점검**: `GET /applications/{신청번호}`, `PATCH /applications/{신청번호}`(기본정보·품목별 인원수·점검완료)
4. `model.py` 랜덤 제거 → 전체 처리 + `POST /optimize/run`(점검완료 대상)
5. `GET /applications`, `GET /schedules/today`
6. `POST /dispatch/confirm`(model2 동선) + `PATCH /schedules/{id}`
7. `PATCH /applications/{신청번호}/complete`
