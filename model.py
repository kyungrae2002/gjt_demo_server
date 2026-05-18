import pandas as pd
import pulp
import numpy as np
import re
from datetime import timedelta
from collections import Counter, defaultdict
import time
import csv
import random


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

    # 랜덤 연속 20개 신청서 추출
    req_dates = df.groupby('신청서번호')['접수일자'].max().sort_values()
    if len(req_dates) > 20:
        start_idx = random.randint(0, len(req_dates) - 20)
        selected_reqs = req_dates.iloc[start_idx : start_idx + 20].index
    else:
        selected_reqs = req_dates.index

    df = df[df['신청서번호'].isin(selected_reqs)].reset_index(drop=True)
    COL_PPL = '필요인원수'

    # ==========================================
    # 2. 근로학생 시간표 로드
    # ==========================================
    avail_base  = {int(k): int(v) for k, v in zip(df_avail['time_index'], df_avail['available_staff'])}
    avail_names = {int(k): str(v) if pd.notna(v) else "" for k, v in zip(df_avail['time_index'], df_avail['names'])}

    # ==========================================
    # 3. 슬롯 달력 생성
    # ==========================================
    ref_date   = df['접수일자'].min()
    ref_monday = ref_date - timedelta(days=ref_date.weekday())
    ref_mon_np = np.datetime64(ref_monday.date())

    def date_to_slot(date):
        d = np.datetime64(pd.to_datetime(date).date())
        return max(1, int(np.busday_count(ref_mon_np, d)) * 7 + 1)

    HOUR_MAP         = [9, 10, 11, 13, 14, 15, 16]
    FORBIDDEN_IDX    = {2, 4, 6}
    SLA_SLOTS        = 5 * 7
    CALENDAR_SLOTS   = 20 * 7
    MAX_STAFF        = 4
    LABOR_PER_PERSON = 3
    K_BY_SLOT        = {0: 36, 1: 30, 3: 21, 5: 9}

    max_r_raw = max(date_to_slot(d) for d in df['접수일자'])
    max_slot  = max_r_raw + CALENDAR_SLOTS

    Avail = {}
    Avail_names = {}
    for s in range(1, max_slot + 1):
        b = ((s - 1) % 35) + 1
        Avail[s] = avail_base[b]
        Avail_names[s] = avail_names[b]

    ALLOWED_T = sorted(
        s for s in range(1, max_slot + 1)
        if Avail[s] > 0 and ((s - 1) % 7) not in FORBIDDEN_IDX
    )

    def slot_to_label(t):
        day_idx     = (t - 1) // 7
        slot_in_day = (t - 1) % 7
        hour        = HOUR_MAP[slot_in_day]
        date_np     = np.busday_offset(ref_mon_np, day_idx)
        date_str    = pd.to_datetime(date_np).strftime('%Y-%m-%d')
        weekday     = ['월','화','수','목','금'][day_idx % 5]
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
            'items':      list(zip(gdf['자산명'], gdf['위치사용명'], p_vals))
        }

    G = list(G_data.keys())
    print(f"[OK] 신청서 {len(G)}개 / 아이템 {len(df)}개 로드 완료")

    # ==========================================
    # 5. 가능한 (g, t) 쌍 사전 필터링
    # ==========================================
    def required_staff_g(g):
        return G_data[g]['max_p_g']

    def can_fit_in_slot(g, t):
        sid = (t - 1) % 7
        if sid not in K_BY_SLOT:
            return False
        max_N = min(MAX_STAFF, Avail[t])
        if required_staff_g(g) > max_N:
            return False
        return G_data[g]['L_g'] <= K_BY_SLOT[sid] * max_N

    feasible_gt = [
        (g, t) for g in G for t in ALLOWED_T
        if t >= G_data[g]['min_r_g']
        and Avail[t] >= required_staff_g(g)
        and can_fit_in_slot(g, t)
    ]
    feasible_T     = sorted(set(t for _, t in feasible_gt))
    feasible_T_set = set(feasible_T)

    g2slots  = {g: [] for g in G}
    t2groups = {t: [] for t in feasible_T}
    for (g, t) in feasible_gt:
        g2slots[g].append(t)
        t2groups[t].append(g)

    print(f"[OK] 가능한 (신청서, 슬롯) 쌍: {len(feasible_gt)}개")

    # ==========================================
    # 6. 파라미터
    # ==========================================
    def _normalize_bldg(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r'^\d+\s*', '', s)
        s = re.sub(r'\s+', '', s)
        return s

    try:
        dist_df = pd.read_excel('datas/건물간_거리_최종.xlsx', index_col=0)
        dist_df.index   = [_normalize_bldg(s) for s in dist_df.index]
        dist_df.columns = [_normalize_bldg(s) for s in dist_df.columns]
        _all_dists = [float(v) for v in dist_df.values.flatten()
                      if pd.notna(v) and float(v) > 0]
        AVG_DIST_M = sum(_all_dists) / len(_all_dists) if _all_dists else 850.0
    except Exception as e:
        print(f"[경고] 엑셀 거리표 로드 실패: {e}")
        dist_df    = pd.DataFrame()
        AVG_DIST_M = 850.0

    DEPT_ALIAS = [
        ('실험연구동A', '실험연구동A'),
        ('실험연구동B', '실험연구동B'),
        ('실험연구',    '실험연구동A'),
        ('공학실험',    '공학실험동'),
        ('공학관',      '공학관'),
        ('체육대학',    '체육대학관'),
        ('외국어대',    '외국어대학관'),
        ('예디대',      '예술디자인대학관'),
        ('예술디자',    '예술디자인대학관'),
        ('원예생명',    '원예생명과학온실'),
        ('생명과학',    '생명과학대학관'),
        ('선승관',      '선승관'),
        ('선공관',      '선승관'),
        ('국제경영',    '국제경영대학관'),
        ('국제학관',    '국제학관'),
        ('학생회관',    '학생회관'),
        ('천문대',      '천문대'),
        ('멀티미디어',  '멀티미디어교육관글로벌관'),
        ('글로벌관',    '멀티미디어교육관글로벌관'),
        ('도예관',      '도예관'),
        ('한방재료',    '한방재료가공'),
        ('한방제조',    '한방재료가공'),
        ('우정원',      '우정원'),
        ('예지원',      '애지원'),
        ('애지원',      '애지원'),
        ('중앙도서',    '중앙도서관'),
        ('대학본부',    '중앙도서관'),
        ('사색',        '사색의광장'),
        ('전자정보',    '전자정보/응용대학관'),
        ('응용과학',    '전자정보/응용대학관'),
        ('응용대학',    '전자정보/응용대학관'),
        ('전정대',      '전자정보/응용대학관'),
        ('제2기숙',     '제2기숙사(남)'),
        ('원자로',      '원자로센터'),
        ('인조잔디',    '인조잔디구장/지하주차장'),
        ('지하주차',    '인조잔디구장/지하주차장'),
        ('평화노천',    '평화노천극장'),
        ('창고',        '창고'),
    ]

    def _bldg_key(dept: str):
        if dept is None:
            return None
        norm = _normalize_bldg(dept)
        if not dist_df.empty and norm in dist_df.index:
            return norm
        for pattern, excel_key in DEPT_ALIAS:
            if pattern in norm:
                return excel_key
        return None

    def get_exact_dist_m(from_dept: str, to_dept: str) -> float:
        b1, b2 = _bldg_key(from_dept), _bldg_key(to_dept)
        if b1 is None or b2 is None or dist_df.empty:
            return AVG_DIST_M
        if b1 == b2:
            return 0.0
        try:
            if b1 in dist_df.index and b2 in dist_df.columns:
                val = dist_df.loc[b1, b2]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass
        try:
            alt_b2 = b2.replace('과학', '공학') if '과학' in b2 else b2.replace('공학', '과학')
            if alt_b2 in dist_df.columns:
                val = dist_df.loc[b1, alt_b2]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass
        return AVG_DIST_M

    CAMPUS_SPEED_MH = 20000
    MGR_PREP_H      = 5/60
    MGR_WAGE        = 10320
    FUEL_PER_L      = 1000
    FUEL_EFF        = 6.5
    OIL_BASE        = 400

    def calc_alpha(dept: str) -> int:
        d_out  = get_exact_dist_m('창고', dept)
        d_back = get_exact_dist_m(dept, '창고')
        rt_km  = (d_out + d_back) / 1000
        fuel   = rt_km / FUEL_EFF * FUEL_PER_L
        mgr_h  = rt_km / (CAMPUS_SPEED_MH / 1000) + MGR_PREP_H
        return int(fuel + MGR_WAGE * mgr_h + OIL_BASE)

    alpha_g = {g: calc_alpha(G_data[g]['dept']) for g in G_data}

    def calc_route_cost(start: str, dest1: str, dest2: str) -> float:
        km   = (get_exact_dist_m(start, dest1)
                + get_exact_dist_m(dest1, dest2)
                + get_exact_dist_m(dest2, start)) / 1000.0
        fuel = km / FUEL_EFF * FUEL_PER_L
        t_h  = km / (CAMPUS_SPEED_MH / 1000)
        return fuel + (t_h + MGR_PREP_H * 2) * MGR_WAGE + OIL_BASE

    optimal_route = {}
    savings_g1_g2 = {}
    pair_feasible = []
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            if i < j:
                common_t = set(g2slots[g1]) & set(g2slots[g2])
                if common_t:
                    d1, d2  = G_data[g1]['dept'], G_data[g2]['dept']
                    cost_AB = calc_route_cost('창고', d1, d2)
                    cost_BA = calc_route_cost('창고', d2, d1)
                    if cost_AB <= cost_BA:
                        cost_bundle             = cost_AB
                        optimal_route[(g1, g2)] = (d1, d2)
                    else:
                        cost_bundle             = cost_BA
                        optimal_route[(g1, g2)] = (d2, d1)
                    savings_g1_g2[(g1, g2)] = max(0, alpha_g[g1] + alpha_g[g2] - cost_bundle)
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
    lp_model = pulp.LpProblem("Logistics_v2", pulp.LpMinimize)

    U        = pulp.LpVariable.dicts("U", feasible_gt, cat='Binary')
    x        = pulp.LpVariable.dicts("x", feasible_T, cat='Binary')
    N        = pulp.LpVariable.dicts("N", feasible_T, lowBound=0, cat='Integer')
    overflow = pulp.LpVariable.dicts("overflow", feasible_T, lowBound=0)
    ov_tier  = {(t, k): pulp.LpVariable(f"ovT_{t}_{k}", lowBound=0)
                for t in feasible_T for k in range(N_TIERS)}
    y_bundle = pulp.LpVariable.dicts("y_bundle", feasible_T, cat='Binary')
    y_pair   = pulp.LpVariable.dicts("y_pair", pair_feasible, cat='Binary')
    M_bundle = max(K_BY_SLOT.values()) * MAX_STAFF + 50
    z_ow     = pulp.LpVariable.dicts("z_ow",  feasible_T, cat='Binary')
    z_ow4    = pulp.LpVariable.dicts("z_ow4", feasible_T, cat='Binary')
    z_ow6    = pulp.LpVariable.dicts("z_ow6", feasible_T, cat='Binary')
    M_ow     = 100

    SRC_9_SLOTS  = [t for t in feasible_T if (t - 1) % 7 == 0]
    SRC_10_SLOTS = [t for t in feasible_T if (t - 1) % 7 == 1]
    z_blk_9_13  = pulp.LpVariable.dicts("z_blk_9_13",  SRC_9_SLOTS,  cat='Binary')
    z_blk_9_15  = pulp.LpVariable.dicts("z_blk_9_15",  SRC_9_SLOTS,  cat='Binary')
    z_blk_10_13 = pulp.LpVariable.dicts("z_blk_10_13", SRC_10_SLOTS, cat='Binary')
    z_blk_10_15 = pulp.LpVariable.dicts("z_blk_10_15", SRC_10_SLOTS, cat='Binary')

    # ==========================================
    # 8. 목적함수
    # ==========================================
    lp_model += (
        pulp.lpSum(alpha_g[g] * U[(g,t)] for (g,t) in feasible_gt) -
        pulp.lpSum(savings_g1_g2[(g1, g2)] * y_pair[(g1, g2, t)] for (g1, g2, t) in pair_feasible) +
        beta   * pulp.lpSum(N[t] for t in feasible_T) +
        w_quad * pulp.lpSum(((t - G_data[g]['min_r_g'])**2) * U[(g,t)] for (g,t) in feasible_gt) +
        pulp.lpSum(OV_TIER_RATES[k] * ov_tier[(t, k)]
                   for t in feasible_T for k in range(N_TIERS))
    ), "Total_Cost"

    # ==========================================
    # 9. 제약조건
    # ==========================================
    external_requests = []
    for g in G:
        slots_g = g2slots[g]
        if not slots_g:
            gd = G_data[g]
            L_max = max(K_BY_SLOT.values()) * MAX_STAFF
            if gd['L_g'] > L_max:
                reason = f"노동량 L={gd['L_g']}이 단일 슬롯 최대치({L_max}) 초과"
            elif gd['max_p_g'] > MAX_STAFF:
                reason = f"필요인원 max_p={gd['max_p_g']}가 MAX_STAFF({MAX_STAFF}) 초과"
            else:
                reason = "당일 완료 가능한 슬롯 없음"
            external_requests.append({'g': g, 'reason': reason, 'data': gd})
            print(f"[외부 위탁] 신청서 {g} ({gd['dept']}): {reason}")
            continue
        lp_model += pulp.lpSum(U[(g, t)] for t in slots_g) == 1, f"Completion_{g}"

    for (g, t) in feasible_gt:
        lp_model += N[t] >= G_data[g]['max_p_g'] * U[(g, t)], f"MinStaff_{g}_{t}"

    for t in feasible_T:
        lp_model += N[t] <= MAX_STAFF * x[t],  f"MaxStaff_{t}"
        lp_model += N[t] <= Avail[t] * x[t],   f"AvailStaff_{t}"

    for t in feasible_T:
        lp_model += x[t] <= pulp.lpSum(U[(g, t)] for g in t2groups[t]), f"NoEmpty_{t}"

    for (g, t) in feasible_gt:
        lp_model += U[(g, t)] <= x[t], f"TripLink_{g}_{t}"

    for t in feasible_T:
        lp_model += pulp.lpSum(U[(g, t)] for g in t2groups[t]) <= 2, f"MaxGroup_{t}"

    for (g1, g2, t) in pair_feasible:
        lp_model += y_pair[(g1, g2, t)] <= U[(g1, t)], f"Pair_Link1_{g1}_{g2}_{t}"
        lp_model += y_pair[(g1, g2, t)] <= U[(g2, t)], f"Pair_Link2_{g1}_{g2}_{t}"
        lp_model += y_pair[(g1, g2, t)] >= U[(g1, t)] + U[(g2, t)] - 1, f"Pair_Link3_{g1}_{g2}_{t}"

    for t in feasible_T:
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += overflow[t] >= L_t - LABOR_PER_PERSON * N[t], f"Overflow_{t}"

    for t in feasible_T:
        lp_model += overflow[t] == pulp.lpSum(ov_tier[(t, k)] for k in range(N_TIERS)), f"OvSum_{t}"
        for k in range(N_TIERS - 1):
            lp_model += ov_tier[(t, k)] <= N[t], f"OvCap_{t}_{k}"

    for t in feasible_T:
        grp_sum = pulp.lpSum(U[(g, t)] for g in t2groups[t])
        lp_model += y_bundle[t] >= grp_sum - 1,     f"BundleFlag_ON_{t}"
        lp_model += y_bundle[t] <= grp_sum / 2,     f"BundleFlag_OFF_{t}"
        L_t_c = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t_c <= 2 * LABOR_PER_PERSON * N[t] + M_bundle * (1 - y_bundle[t]), f"BundleCap_{t}"

    for t in feasible_T:
        lp_model += z_ow[t] <= x[t], f"OW_notrip_{t}"
        lp_model += overflow[t] <= LABOR_PER_PERSON * N[t] - 1 + M_ow * z_ow[t] + M_ow * (1 - x[t]), f"OW_flag_{t}"

    for t in feasible_T:
        sid = (t - 1) % 7
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t <= K_BY_SLOT[sid] * N[t], f"SlotCap_{t}"

    LUNCH_TRANS = {0: (15, 3), 1: (9, 2)}
    for t in feasible_T:
        sid = (t - 1) % 7
        if sid not in LUNCH_TRANS:
            continue
        pre_cap, offset = LUNCH_TRANS[sid]
        t_13 = t + offset
        L_t  = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        if t_13 in feasible_T_set:
            lp_model += (L_t <= pre_cap * N[t] + LABOR_PER_PERSON * 7 * (Avail[t_13] - N[t_13])), f"AfterLunchCap_{t}"
        else:
            lp_model += L_t <= pre_cap * N[t], f"AfterLunchCap_{t}"

    SUB_BLOCKS_PER_HOUR = [2, 2, 1, 2, 2, 2, 1]
    work_pairs = []
    s_to_hrs   = defaultdict(list)
    hr_to_srcs = defaultdict(list)
    for s in feasible_T:
        day_s = (s - 1) // 7
        for offset in range(7):
            hr = s + offset
            if (hr - 1) // 7 != day_s:
                break
            sid_hr = (hr - 1) % 7
            if SUB_BLOCKS_PER_HOUR[sid_hr] == 0 or Avail.get(hr, 0) == 0:
                continue
            work_pairs.append((s, hr))
            s_to_hrs[s].append(hr)
            hr_to_srcs[hr].append(s)

    w = pulp.LpVariable.dicts("w", work_pairs, lowBound=0)

    for s in feasible_T:
        L_s = pulp.lpSum(G_data[g]['L_g'] * U[(g, s)] for g in t2groups[s])
        if s_to_hrs[s]:
            lp_model += pulp.lpSum(w[(s, hr)] for hr in s_to_hrs[s]) == L_s, f"WorkSum_{s}"

    for hr, srcs in hr_to_srcs.items():
        sid_hr = (hr - 1) % 7
        cap = Avail[hr] * SUB_BLOCKS_PER_HOUR[sid_hr] * LABOR_PER_PERSON
        lp_model += pulp.lpSum(w[(s, hr)] for s in srcs) <= cap, f"HourPool_{hr}"

    for (s, hr) in work_pairs:
        sid_hr = (hr - 1) % 7
        lp_model += w[(s, hr)] <= N[s] * SUB_BLOCKS_PER_HOUR[sid_hr] * LABOR_PER_PERSON

    for t in feasible_T:
        lp_model += z_ow4[t] <= x[t], f"OW4_notrip_{t}"
        lp_model += overflow[t] <= 3 * LABOR_PER_PERSON * N[t] - 1 + M_ow * z_ow4[t] + M_ow * (1 - x[t]), f"OW4_flag_{t}"

    for t in feasible_T:
        lp_model += z_ow6[t] <= x[t], f"OW6_notrip_{t}"
        lp_model += overflow[t] <= 5 * LABOR_PER_PERSON * N[t] - 1 + M_ow * z_ow6[t] + M_ow * (1 - x[t]), f"OW6_flag_{t}"

    def _same_day(t1, t2):
        return (t1 - 1) // 7 == (t2 - 1) // 7

    for t in feasible_T:
        nxt = t + 1
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_ow[t] <= 1, f"NextSlotBlk_{t}"

    for t in feasible_T:
        nxt = t + 2
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_ow4[t] <= 1, f"NextNextBlk_{t}"

    for t in feasible_T:
        nxt = t + 3
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_ow6[t] <= 1, f"NextNextNextBlk_{t}"

    for t in SRC_9_SLOTS:
        lp_model += z_blk_9_13[t] <= x[t], f"Blk9_13_notrip_{t}"
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t <= 15 * N[t] - 1 + M_ow * z_blk_9_13[t] + M_ow * (1 - x[t]), f"Blk9_13_flag_{t}"
        nxt = t + 3
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_blk_9_13[t] <= 1, f"Blk9_13_link_{t}"

    for t in SRC_9_SLOTS:
        lp_model += z_blk_9_15[t] <= x[t], f"Blk9_15_notrip_{t}"
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t <= 27 * N[t] - 1 + M_ow * z_blk_9_15[t] + M_ow * (1 - x[t]), f"Blk9_15_flag_{t}"
        nxt = t + 5
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_blk_9_15[t] <= 1, f"Blk9_15_link_{t}"

    for t in SRC_10_SLOTS:
        lp_model += z_blk_10_13[t] <= x[t], f"Blk10_13_notrip_{t}"
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t <= 9 * N[t] - 1 + M_ow * z_blk_10_13[t] + M_ow * (1 - x[t]), f"Blk10_13_flag_{t}"
        nxt = t + 2
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_blk_10_13[t] <= 1, f"Blk10_13_link_{t}"

    for t in SRC_10_SLOTS:
        lp_model += z_blk_10_15[t] <= x[t], f"Blk10_15_notrip_{t}"
        L_t = pulp.lpSum(G_data[g]['L_g'] * U[(g, t)] for g in t2groups[t])
        lp_model += L_t <= 21 * N[t] - 1 + M_ow * z_blk_10_15[t] + M_ow * (1 - x[t]), f"Blk10_15_flag_{t}"
        nxt = t + 4
        if nxt in feasible_T_set and _same_day(t, nxt):
            lp_model += x[nxt] + z_blk_10_15[t] <= 1, f"Blk10_15_link_{t}"

    # ==========================================
    # 10. 풀이
    # ==========================================
    print(f"\n[START] 최적화 실행 중...")
    t0 = time.time()
    lp_model.solve(pulp.PULP_CBC_CMD(timeLimit=300, msg=0, gapRel=0.05))
    elapsed = time.time() - t0

    print(f">> 소요 시간   : {elapsed:.1f}초")
    print(f">> 최적화 상태 : {pulp.LpStatus[lp_model.status]}")
    if pulp.value(lp_model.objective):
        print(f">> 최소 비용   : {int(pulp.value(lp_model.objective)):,}원")

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
        actual_N_csv   = int(round(pulp.value(N[t]))) if pulp.value(N[t]) else 0
        names_csv      = Avail_names[t] if Avail_names[t] else "-"

        for g in assigned_g:
            gd = G_data[g]
            item_cnt = Counter()
            for item_name, loc, ppl in gd['items']:
                item_cnt[(item_name, loc, ppl)] += 1
            for (item_name, loc, ppl), cnt in item_cnt.items():
                output_rows.append({
                    '출동일시':   dispatch_label,
                    '신청서번호': g,
                    '신청부서':   gd['dept'],
                    '설치장소':   loc,
                    '품명':       item_name,
                    '수량':       cnt,
                    '가용명단':   names_csv,
                    '투입인원수': actual_N_csv,
                })

    if external_requests:
        print(f"\n[외부 위탁 필요] 총 {len(external_requests)}건")
        for entry in external_requests:
            print(f"  - 신청서 {entry['g']} | {entry['data']['dept']} | {entry['reason']}")

    return output_rows


if __name__ == "__main__":
    df       = pd.read_csv("datas/불용신청_생성데이터.csv", encoding='utf-8-sig')
    df_avail = pd.read_csv("datas/근로학생시간.csv")

    results = run_optimizer(df, df_avail)

    output_path = "datas/출동결과.csv"
    cols = ['출동일시', '신청서번호', '신청부서', '설치장소', '품명', '수량', '가용명단', '투입인원수']
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] 출동 결과 CSV 저장 완료 ({len(results)}행): {output_path}")
