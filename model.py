import os
import platform
import pandas as pd
import pulp
import numpy as np
from datetime import timedelta, timezone
import time
import csv
from collections import defaultdict


def _get_cbc_solver(timeLimit=120, msg=0, gapRel=0.05):
    """OS에 맞는 CBC 솔버를 반환한다.

    macOS(특히 Apple Silicon)에서는 PuLP 번들 CBC가 인텔 전용 바이너리라
    'Bad CPU type' 에러가 난다. 이때는 Homebrew로 설치한 네이티브 CBC를 쓴다.
    Linux 서버 등에서는 번들 CBC가 정상 동작하므로 기본값을 그대로 사용한다.
    CBC_PATH 환경변수로 경로를 직접 지정할 수도 있다.
    """
    if platform.system() == "Darwin":
        cbc_path = os.getenv("CBC_PATH", "/opt/homebrew/bin/cbc")
        if os.path.exists(cbc_path):
            return pulp.COIN_CMD(path=cbc_path, timeLimit=timeLimit, msg=msg, gapRel=gapRel)
    return pulp.PULP_CBC_CMD(timeLimit=timeLimit, msg=msg, gapRel=gapRel)


def run_optimizer(df, df_avail):
    # ==========================================
    # 1. 데이터 전처리
    # ==========================================
    df = df.rename(columns={
        '품명': '자산명', '설치장소': '위치사용명',
        '신청번호': '신청서번호', '신청일자': '접수일자', '신청부서': '신청조직'
    })
    if '수량' in df.columns:
        df = df[df['수량'] > 0]
        df = df.loc[df.index.repeat(df['수량'].astype(int))].assign(수량=1).reset_index(drop=True)
    df['접수일자'] = pd.to_datetime(df['접수일자'])
    df = df.sort_values('접수일자')

    # 전달된 신청서 전체를 대상으로 최적화한다.
    # (재최적화 시 미완료 신청서 전체를 합쳐 넣어 출동 시간 충돌을 해소하려는 목적)
    df = df.reset_index(drop=True)
    COL_PPL = '필요인원수'

    # 스케줄까지 흘려보낼 선택 필드 (신청서에서 온 경우 존재, 없으면 빈 값)
    for _c in ('자산번호', '규격모델', '금액'):
        if _c not in df.columns:
            df[_c] = ''

    # ==========================================
    # 2. 근로학생 시간표 로드
    # ==========================================
    avail_base  = {int(k): int(v) for k, v in zip(df_avail['time_index'], df_avail['available_staff'])}
    avail_names = {int(k): str(v) if pd.notna(v) else "" for k, v in zip(df_avail['time_index'], df_avail['names'])}

    # ==========================================
    # 3. 슬롯 달력 생성
    # ==========================================
    # Q6: 출동일시 기준일 = 실행일(오늘). 과거 슬롯이 생기지 않게 한다.
    # 서버(UTC)가 아닌 한국 시각 기준 오늘. 고정 오프셋 +9(서머타임 없음)로 안전.
    ref_date   = pd.Timestamp.now(tz=timezone(timedelta(hours=9))).tz_localize(None).normalize()
    ref_monday = ref_date - timedelta(days=ref_date.weekday())
    ref_mon_np = np.datetime64(ref_monday.date())

    def date_to_slot(date):
        d = np.datetime64(pd.to_datetime(date).date())
        return max(1, int(np.busday_count(ref_mon_np, d)) * 7 + 1)

    HOUR_MAP        = [9, 10, 11, 13, 14, 15, 16]
    FORBIDDEN_IDX   = {2, 4, 6}
    SLA_SLOTS       = 5 * 7
    MAX_STAFF       = 4
    LABOR_PER_PERSON = 3

    max_r_raw = max(date_to_slot(d) for d in df['접수일자'])
    max_slot  = max_r_raw + SLA_SLOTS

    Avail = {}; Avail_names = {}
    for s in range(1, max_slot + 1):
        b = ((s - 1) % 35) + 1
        Avail[s] = avail_base[b]; Avail_names[s] = avail_names[b]

    ALLOWED_T = sorted(
        s for s in range(1, max_slot + 1)
        if Avail[s] > 0 and ((s - 1) % 7) not in FORBIDDEN_IDX
    )

    def slot_to_label(t):
        day_idx      = (t - 1) // 7
        slot_in_day  = (t - 1) % 7
        hour         = HOUR_MAP[slot_in_day]
        date_np      = np.busday_offset(ref_mon_np, day_idx)
        date_str     = pd.to_datetime(date_np).strftime('%Y-%m-%d')
        weekday      = ['월','화','수','목','금'][day_idx % 5]
        return f"{date_str} ({weekday}) {hour:02d}:00"

    # ==========================================
    # 4. 신청서(그룹) 단위 사전 계산
    # ==========================================
    G_data = {}
    for req_id, gdf in df.groupby('신청서번호'):
        p_vals = [min(int(r), 4) for r in gdf[COL_PPL]]
        G_data[str(req_id)] = {
            'L_g':        sum(p_vals),
            'max_p_g':    max(p_vals),
            'min_r_g':    date_to_slot(gdf['접수일자'].min()),
            'deadline_g': date_to_slot(gdf['접수일자'].min()) + SLA_SLOTS - 1,
            'dept':       gdf['신청조직'].iloc[0],
            'recv':       gdf['접수일자'].min().strftime('%Y-%m-%d'),
            'items':      [
                {'품명': n, '설치장소': l, '필요인원수': p,
                 '자산번호': a, '규격모델': s, '금액': m}
                for n, l, p, a, s, m in zip(
                    gdf['자산명'], gdf['위치사용명'], p_vals,
                    gdf['자산번호'], gdf['규격모델'], gdf['금액'])
            ]
        }

    G = list(G_data.keys())
    print(f"[OK] 신청서 {len(G)}개 / 아이템 {len(df)}개 로드 완료")

    # ==========================================
    # 5. 가능한 (g, t) 쌍 사전 필터링
    # ==========================================
    feasible_gt = [
        (g, t) for g in G for t in ALLOWED_T
        if t >= G_data[g]['min_r_g']
        and Avail[t] >= G_data[g]['max_p_g']
    ]
    feasible_T = sorted(set(t for _, t in feasible_gt))

    g2slots  = {g: [] for g in G}
    t2groups = {t: [] for t in feasible_T}
    for (g, t) in feasible_gt:
        g2slots[g].append(t)
        t2groups[t].append(g)

    print(f"[OK] 가능한 (신청서, 슬롯) 쌍: {len(feasible_gt)}개")

    # ==========================================
    # 6. 파라미터
    # ==========================================
    DEPT_TO_NODE = {
        '예지원': 2, '우정원': 4, '공학관': 6, '공학실험': 8, '체육대학': 9, '외국어대': 10, '제2기숙': 13,
        '도예관': 15, '선승관': 17, '선공관': 17, '생명과학': 18, '한방제조': 19, '실험연구': 20,
        '예술디자': 21, '예디대': 21, '국제경영': 22, '학생회관': 23, '중앙도서': 24, '대학본부': 24,
        '사색': 25, '전자정보': 31, '응용과학': 31, '국제학관': 32, '천문대': 33, '창고': 34
    }
    try:
        dist_df = pd.read_excel(r'datas/건물간 거리_창고 추가.xlsx', index_col=0)
        _all_dists = [float(dist_df.loc[i, c]) for i in dist_df.index for c in dist_df.columns if pd.notna(dist_df.loc[i, c]) and float(dist_df.loc[i, c]) > 0]
        AVG_DIST_M = sum(_all_dists) / len(_all_dists) if _all_dists else 750.0
    except Exception as e:
        print(f"[경고] 엑셀 거리표 로드 실패: {e}")
        dist_df = pd.DataFrame()
        AVG_DIST_M = 750.0

    def get_exact_dist_m(dept1, dept2):
        n1 = next((v for k, v in DEPT_TO_NODE.items() if k in dept1), None)
        n2 = next((v for k, v in DEPT_TO_NODE.items() if k in dept2), None)
        if n1 is None or n2 is None or dist_df.empty:
            return AVG_DIST_M
        if n1 == n2:
            return 0.0
        val = dist_df.loc[n1, n2] if n2 in dist_df.columns else np.nan
        if pd.isna(val):
            val = dist_df.loc[n2, n1] if n1 in dist_df.columns else np.nan
        return float(val) if pd.notna(val) else AVG_DIST_M

    def dist_m(dept):
        return get_exact_dist_m('창고', dept)

    CAMPUS_SPEED_MH  = 20000
    MGR_PREP_H       = 5/60
    MGR_WAGE         = 10320
    FUEL_PER_L       = 1000
    FUEL_EFF         = 6.5
    OIL_BASE         = 400

    def calc_alpha(dept):
        d      = dist_m(dept)
        rt_km  = (d * 2) / 1000
        fuel   = rt_km / FUEL_EFF * FUEL_PER_L
        mgr_h  = rt_km / (CAMPUS_SPEED_MH / 1000) + MGR_PREP_H
        return int(fuel + MGR_WAGE * mgr_h + OIL_BASE)

    alpha_g = {g: calc_alpha(G_data[g]['dept']) for g in G_data}

    def calc_bundled_cost(dept1, dept2):
        tot_dist_km = (dist_m(dept1) + get_exact_dist_m(dept1, dept2) + dist_m(dept2)) / 1000.0
        fuel_cost = tot_dist_km / FUEL_EFF * FUEL_PER_L
        travel_h  = tot_dist_km / (CAMPUS_SPEED_MH/1000)
        return fuel_cost + (travel_h + MGR_PREP_H * 2) * MGR_WAGE + OIL_BASE

    savings_g1_g2 = {}
    pair_feasible = []
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            if i < j:
                common_t = set(g2slots[g1]) & set(g2slots[g2])
                if common_t:
                    cost_single = alpha_g[g1] + alpha_g[g2]
                    cost_bundle = calc_bundled_cost(G_data[g1]['dept'], G_data[g2]['dept'])
                    savings_g1_g2[(g1, g2)] = max(0, cost_single - cost_bundle)
                    for t in common_t:
                        pair_feasible.append((g1, g2, t))

    alpha_avg = int(sum(alpha_g.values()) / len(alpha_g))
    print(f"[OK] alpha 평균: {alpha_avg:,}원")

    beta   = 5160
    w_quad = alpha_avg / (35**2)

    N_TIERS       = 10
    OV_TIER_RATES = [1868, 2044, 2257, 2518, 2849, 3279, 3860, 4694, 5993, 8270]

    # ==========================================
    # 7. 결정 변수
    # ==========================================
    model = pulp.LpProblem("Logistics_v2", pulp.LpMinimize)

    U        = pulp.LpVariable.dicts("U", feasible_gt, cat='Binary')
    x        = pulp.LpVariable.dicts("x", feasible_T, cat='Binary')
    N        = pulp.LpVariable.dicts("N", feasible_T, lowBound=0, cat='Integer')
    overflow = pulp.LpVariable.dicts("overflow", feasible_T, lowBound=0)
    ov_tier  = {(t, k): pulp.LpVariable(f"ovT_{t}_{k}", lowBound=0)
                for t in feasible_T for k in range(N_TIERS)}
    y_bundle = pulp.LpVariable.dicts("y_bundle", feasible_T, cat='Binary')
    y_pair   = pulp.LpVariable.dicts("y_pair", pair_feasible, cat='Binary')
    M_bundle = max(sum(G_data[g]['L_g'] for g in G_data) // len(G_data) * 3 + 10, 50)
    z_ow     = pulp.LpVariable.dicts("z_ow", feasible_T, cat='Binary')
    M_ow     = 60

    session_slots = defaultdict(list)
    for t in feasible_T:
        day  = t // len(HOUR_MAP)
        hi   = t  % len(HOUR_MAP)
        sess = 'AM' if hi < 3 else 'PM'
        session_slots[(day, sess)].append(t)

    # ==========================================
    # 8. 목적함수
    # ==========================================
    model += (
        pulp.lpSum(alpha_g[g] * U[(g,t)] for (g,t) in feasible_gt) -
        pulp.lpSum(savings_g1_g2[(g1, g2)] * y_pair[(g1, g2, t)] for (g1, g2, t) in pair_feasible) +
        beta  * pulp.lpSum(N[t] for t in feasible_T) +
        w_quad * pulp.lpSum(((t - G_data[g]['min_r_g'])**2) * U[(g,t)] for (g,t) in feasible_gt) +
        pulp.lpSum(OV_TIER_RATES[k] * ov_tier[(t, k)] for t in feasible_T for k in range(N_TIERS))
    ), "Total_Cost"

    # ==========================================
    # 9. 제약조건
    # ==========================================
    for g in G:
        slots_g = g2slots[g]
        if not slots_g:
            print(f"[경고] 신청서 {g}: 가능한 슬롯 없음")
            continue
        model += pulp.lpSum(U[(g, t)] for t in slots_g) == 1, f"Completion_{g}"

    for (g, t) in feasible_gt:
        model += N[t] >= G_data[g]['max_p_g'] * U[(g, t)], f"MinStaff_{g}_{t}"

    HEAVY_L_THRESHOLD = 18
    for g in G:
        if G_data[g]['L_g'] >= HEAVY_L_THRESHOLD:
            for t in g2slots[g]:
                model += N[t] >= MAX_STAFF * U[(g, t)], f"HeavyForce_{g}_{t}"

    for t in feasible_T:
        model += N[t] <= MAX_STAFF * x[t],  f"MaxStaff_{t}"
        model += N[t] <= Avail[t] * x[t],   f"AvailStaff_{t}"

    for t in feasible_T:
        model += x[t] <= pulp.lpSum(U[(g, t)] for g in t2groups[t]), f"NoEmpty_{t}"

    for (g, t) in feasible_gt:
        model += U[(g, t)] <= x[t], f"TripLink_{g}_{t}"

    for t in feasible_T:
        model += pulp.lpSum(U[(g, t)] for g in t2groups[t]) <= 2, f"MaxGroup_{t}"

    for (g1, g2, t) in pair_feasible:
        model += y_pair[(g1, g2, t)] <= U[(g1, t)], f"Pair_Link1_{g1}_{g2}_{t}"
        model += y_pair[(g1, g2, t)] <= U[(g2, t)], f"Pair_Link2_{g1}_{g2}_{t}"
        model += y_pair[(g1, g2, t)] >= U[(g1, t)] + U[(g2, t)] - 1, f"Pair_Link3_{g1}_{g2}_{t}"

    for t in feasible_T:
        grp_sum   = pulp.lpSum(U[(g, t)] for g in t2groups[t])
        model += y_bundle[t] >= (grp_sum - 1) / 2, f"BundleFlag_{t}"
        L_t_expr_c = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        model += L_t_expr_c <= 2 * LABOR_PER_PERSON * N[t] + M_bundle * (1 - y_bundle[t]), f"BundleCap_{t}"

    for t in feasible_T:
        L_t_expr = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        model += overflow[t] >= L_t_expr - LABOR_PER_PERSON * N[t], f"Overflow_{t}"

    for t in feasible_T:
        model += overflow[t] == pulp.lpSum(ov_tier[(t, k)] for k in range(N_TIERS)), f"OvSum_{t}"
        for k in range(N_TIERS - 1):
            model += ov_tier[(t, k)] <= N[t], f"OvCap_{t}_{k}"

    for t in feasible_T:
        model += z_ow[t] <= x[t], f"OW_notrip_{t}"
        model += overflow[t] <= LABOR_PER_PERSON * N[t] + M_ow * z_ow[t], f"OW_flag_{t}"

    for (day, sess), slots in session_slots.items():
        feas = [t for t in slots if t in set(feasible_T)]
        for i, t1 in enumerate(feas):
            for t2 in feas[i+1:]:
                model += x[t1] + z_ow[t2] <= 1, f"SessBlk_{t2}_{t1}"
                model += x[t2] + z_ow[t1] <= 1, f"SessBlk_{t1}_{t2}"

    # ==========================================
    # 10. 풀이
    # ==========================================
    print(f"\n[START] 최적화 실행 중...")
    t0 = time.time()
    model.solve(_get_cbc_solver(timeLimit=120, msg=0, gapRel=0.05))
    elapsed = time.time() - t0

    print(f">> 소요 시간   : {elapsed:.1f}초")
    print(f">> 최적화 상태 : {pulp.LpStatus[model.status]}")
    if pulp.value(model.objective):
        print(f">> 최소 비용   : {int(pulp.value(model.objective)):,}원")

    # ==========================================
    # 11. 결과 수집
    # ==========================================
    output_rows = []
    for t in feasible_T:
        if pulp.value(x[t]) is None or pulp.value(x[t]) < 0.5:
            continue
        assigned_g = [g for g in t2groups[t]
                      if pulp.value(U[(g,t)]) is not None and pulp.value(U[(g,t)]) > 0.5]
        if not assigned_g:
            continue

        dispatch_label = slot_to_label(t)
        staff_t = int(round(pulp.value(N[t]) or 0))   # 이 슬롯 최적화가 정한 투입 인원(N[t])
        names_t = Avail_names.get(t, "")               # 그 슬롯 가용 근로학생 명단
        for g in assigned_g:
            gd = G_data[g]
            for item in gd['items']:
                output_rows.append({
                    '출동일시':   dispatch_label,
                    '신청번호':   g,                 # 연결 키 (applications.신청번호)
                    '신청일자':   gd['recv'],         # 접수일자 (YYYY-MM-DD)
                    '자산번호':   item['자산번호'],
                    '품명':       item['품명'],
                    '규격모델':   item['규격모델'],
                    '설치장소':   item['설치장소'],
                    '신청부서':   gd['dept'],
                    '수량':       1,
                    '금액':       item['금액'],
                    '필요인원수': item['필요인원수'],
                    '투입인원수': staff_t,
                    '가용명단':   names_t,
                })

    return output_rows


if __name__ == "__main__":
    df       = pd.read_csv("datas/불용신청_생성데이터.csv", encoding='utf-8-sig')
    df_avail = pd.read_csv("datas/근로학생시간.csv")

    results = run_optimizer(df, df_avail)

    output_path = "datas/출동결과.csv"
    cols = ['출동일시', '품명', '설치장소', '신청부서', '수량', '필요인원수']
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] 출동 결과 CSV 저장 완료 ({len(results)}행): {output_path}")
