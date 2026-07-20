# indoor_route.py

import itertools
import heapq
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd


PARAMS = {
    "input_json_path": "schedule.json",
    "nodes_edges_dir": "nodes_edges",
    "default_dispatch_people": 2,
    "exact_tsp_threshold": 8,
    "walk_cost_per_meter": 1.0,
    "stair_cost": 100.0,
    "elevator_cost": 100.0,
    "return_to_start": True,
    "output_dir": "route_outputs",
}


def read_csv_safely(path):
    path = Path(path)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path)


def read_json_safely(path):
    path = Path(path)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            pass
    with open(path, "r") as f:
        return json.load(f)


def normalize_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def normalize_floor(value):
    s = str(value).strip().upper()
    if s.endswith("F"):
        return s
    if s.endswith("층"):
        return s.replace("층", "F")
    try:
        return f"{int(float(s))}F"
    except Exception:
        return s


def floor_to_int(value):
    """
    노드의 층 값을 정수로 변환한다.
    예: 5F, 5층, 5.0 -> 5
    변환이 불가능하면 None을 반환한다.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()
    text = text.replace("층", "F")

    match = re.search(r"-?\d+", text)
    if not match:
        return None

    return int(match.group())


def get_node_floor_num(G, node):
    """
    그래프 노드 속성에서 floor/층 컬럼을 찾아 층 번호를 반환한다.
    nodes CSV의 층 컬럼명이 floor, 층, floor_num 등이어도 최대한 대응한다.
    """
    data = G.nodes[node]

    for key in ["floor", "층", "floor_num", "Floor", "FLOOR"]:
        if key in data:
            return floor_to_int(data.get(key))

    for key, value in data.items():
        key_text = str(key).lower()
        if "floor" in key_text or "층" in key_text:
            return floor_to_int(value)

    return None


def get_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"필수 컬럼이 없습니다. 후보={candidates}, 실제 컬럼={list(df.columns)}")
    return None


def to_float(value, default):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return float(default)
    return float(converted)


def normalize_edge_type(value):
    text = str(value).strip().lower()
    if text in {"walk", "corridor", "hallway", "복도", "통로", "보행"}:
        return "walk"
    if text in {"stair", "stairs", "계단"}:
        return "stair"
    if text in {"elevator", "lift", "엘리베이터", "엘베"}:
        return "elevator"
    return text if text else "other"


def split_location_to_building_room(location):
    """설치장소에서 건물명(한글)만 파싱한다. 예:
    '공학관 401호'                     -> ('공학관', '401호')
    '도서관102호 (학술연구지원팀)'      -> ('도서관', '102호')
    '도서관B101호 (자료열람실)'         -> ('도서관', 'B101호')
    '중앙도서관'                        -> ('중앙도서관', None)
    """
    if location is None or pd.isna(location):
        return "", None
    text = str(location).strip()
    if not text:
        return "", None
    # 1) 뒤쪽 괄호 설명(예: '(학술연구지원팀)') 제거
    text = re.sub(r"\s*\(.*\)\s*$", "", text).strip()
    # 2) 끝의 호실(예: '401호', 'B101호', 'B101B호', '102') 제거 → 건물명만 남김
    building = re.sub(r"\s*[A-Za-z]?\d+[A-Za-z0-9]*호?\s*$", "", text).strip()
    if not building:  # 전부 호실로 인식된 예외 상황 방지
        building = text
    room = text[len(building):].strip() or None
    return building, room


def load_json_schedule(input_json_path, params=PARAMS):
    data = read_json_safely(input_json_path)
    dispatch_people = to_float(data.get("투입인원수"), params["default_dispatch_people"])
    requests = data.get("신청서", [])

    if not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("JSON 입력에 '신청서' 리스트가 없거나 비어 있습니다.")

    rows = []
    for idx, item in enumerate(requests, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"신청서의 {idx}번째 항목이 객체가 아닙니다: {item}")

        building, room = split_location_to_building_room(item.get("설치장소", ""))

        row = dict(item)
        row["_input_row_id"] = idx
        row["_building"] = building
        row["_room"] = room
        row["_quantity"] = to_float(item.get("수량", 1), 1)
        row["_required_people"] = to_float(item.get("필요인원수", 1), 1)
        row["_dispatch_people"] = dispatch_people
        rows.append(row)

    schedule = pd.DataFrame(rows)

    if schedule["_building"].eq("").any():
        raise ValueError("설치장소에서 건물명을 추출하지 못한 항목이 있습니다.")

    return schedule, dispatch_people, {"raw_input": data}


def split_assigned_rooms(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[;,/|]", text)
    return [p.strip() for p in parts if p.strip()]


def build_room_to_node(nodes):
    node_col = get_col(nodes, ["node_id"])
    assigned_col = get_col(nodes, ["assigned_rooms", "assigned_room", "room", "rooms"], required=False)

    if assigned_col is None:
        raise ValueError("nodes 파일에 호수 매핑 컬럼이 없습니다. assigned_rooms 컬럼을 확인하세요.")

    mapping = {}
    for _, row in nodes.iterrows():
        node_id = str(row[node_col]).strip()
        for room in split_assigned_rooms(row[assigned_col]):
            mapping[normalize_text(room)] = node_id
    return mapping



def room_to_number(room):
    """
    호수 문자열을 비교 가능한 숫자로 변환한다.
    예: 329 -> 329, 329-1 -> 329.1, B03 -> -3
    변환 불가능하면 None을 반환한다.
    """
    if room is None or pd.isna(room):
        return None

    text = str(room).strip().upper()
    if not text:
        return None

    # 지하 호수는 B03처럼 들어오는 경우가 있어 음수로 취급한다.
    basement = text.startswith("B")

    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None

    main = int(numbers[0])
    decimal = 0.0
    if len(numbers) >= 2:
        # 329-1 같은 세부 호수는 329.1처럼 약하게 반영한다.
        decimal = int(numbers[1]) / (10 ** len(numbers[1]))

    value = main + decimal
    return -value if basement else value


def infer_floor_from_room(room):
    """
    호수에서 층을 추정한다.
    예: 329 -> 3층, 566 -> 5층, B03 -> 0층
    """
    if room is None or pd.isna(room):
        return None

    text = str(room).strip().upper()
    if not text:
        return None

    if text.startswith("B"):
        return 0

    match = re.search(r"\d+", text)
    if not match:
        return None

    number_text = match.group()
    if len(number_text) >= 3:
        return int(number_text[0])
    if len(number_text) == 2:
        return int(number_text[0])
    return None


def build_room_candidates(nodes):
    """
    nodes의 assigned_rooms를 펼쳐서 대체 매핑 후보 목록을 만든다.
    """
    node_col = get_col(nodes, ["node_id"])
    assigned_col = get_col(nodes, ["assigned_rooms", "assigned_room", "room", "rooms"], required=False)
    floor_col = get_col(nodes, ["floor", "층", "floor_num"], required=False)

    if assigned_col is None:
        raise ValueError("nodes 파일에 호수 매핑 컬럼이 없습니다. assigned_rooms 컬럼을 확인하세요.")

    candidates = []

    for _, row in nodes.iterrows():
        node_id = str(row[node_col]).strip()
        node_floor = floor_to_int(row[floor_col]) if floor_col is not None else None

        for room in split_assigned_rooms(row[assigned_col]):
            room_number = room_to_number(room)
            if room_number is None:
                continue

            candidates.append({
                "node": node_id,
                "room": str(room),
                "room_key": normalize_text(room),
                "room_number": room_number,
                "room_floor": infer_floor_from_room(room),
                "node_floor": node_floor,
            })

    return candidates


def find_nearest_room_node(request_room, nodes):
    """
    요청 호수가 nodes에 정확히 없을 때 가장 가까운 호수를 가진 노드를 찾는다.

    우선순위:
    1) 요청 호수에서 추정한 층과 같은 층의 후보 우선
    2) 호수 숫자 차이가 가장 작은 후보
    3) 같은 차이면 node_id, room 문자열 기준으로 안정적으로 선택
    """
    request_number = room_to_number(request_room)
    request_floor = infer_floor_from_room(request_room)

    if request_number is None:
        return None

    candidates = build_room_candidates(nodes)
    if not candidates:
        return None

    def score(candidate):
        candidate_floor = candidate.get("node_floor")
        if candidate_floor is None:
            candidate_floor = candidate.get("room_floor")

        same_floor_penalty = 0
        if request_floor is not None and candidate_floor is not None:
            same_floor_penalty = 0 if request_floor == candidate_floor else 1

        return (
            same_floor_penalty,
            abs(candidate["room_number"] - request_number),
            str(candidate["node"]),
            str(candidate["room"]),
        )

    best = min(candidates, key=score)
    return {
        "node": best["node"],
        "nearest_room": best["room"],
        "distance": abs(best["room_number"] - request_number),
        "request_room": str(request_room),
        "request_floor": request_floor,
        "nearest_floor": best.get("node_floor") if best.get("node_floor") is not None else best.get("room_floor"),
    }


def get_start_nodes(nodes):
    node_col = get_col(nodes, ["node_id"])
    start_col = get_col(nodes, ["start_node", "is_start", "start"], required=False)

    if start_col is None:
        raise ValueError("nodes 파일에 start_node 컬럼이 없습니다.")

    starts = []
    for _, row in nodes.iterrows():
        value = row[start_col]
        if pd.isna(value):
            continue
        if str(value).strip() in {"1", "1.0", "True", "true", "Y", "y", "yes", "YES"}:
            starts.append(str(row[node_col]).strip())

    if not starts:
        raise ValueError("start_node=1인 시작 노드가 없습니다.")

    return starts


def map_items_to_nodes(building_items, nodes):
    room_to_node = build_room_to_node(nodes)
    items = building_items.copy()

    items["_room_key"] = items["_room"].apply(normalize_text)
    items["_node"] = items["_room_key"].map(room_to_node)
    items["_mapping_type"] = items["_node"].apply(lambda x: "exact" if pd.notna(x) else None)
    items["_mapped_room"] = items["_room"]
    items["_nearest_room"] = None
    items["_nearest_room_distance"] = None

    # 정확히 매핑되는 호수가 없으면, 가장 가까운 호수를 가진 노드로 대체 매핑한다.
    # 단, 안내 문구에는 _room, 즉 원래 신청 호수를 그대로 사용한다.
    for idx, row in items[items["_node"].isna()].iterrows():
        fallback = find_nearest_room_node(row.get("_room"), nodes)
        if fallback is None:
            continue

        items.at[idx, "_node"] = fallback["node"]
        items.at[idx, "_mapping_type"] = "nearest_room"
        items.at[idx, "_mapped_room"] = row.get("_room")
        items.at[idx, "_nearest_room"] = fallback["nearest_room"]
        items.at[idx, "_nearest_room_distance"] = fallback["distance"]

    unmatched = items[items["_node"].isna()].copy()
    matched = items.dropna(subset=["_node"]).copy()

    grouped = (
        matched.groupby("_node")
        .agg(
            # rooms는 실제 안내해야 하는 신청 호수 기준으로 유지한다.
            rooms=("_room", lambda x: sorted(set(str(v) for v in x if pd.notna(v)))),
            request_count=("_room", "size"),
            total_quantity=("_quantity", "sum"),
            max_required_people=("_required_people", "max"),
            input_row_ids=("_input_row_id", list),
            mapping_types=("_mapping_type", lambda x: sorted(set(str(v) for v in x if pd.notna(v)))),
            nearest_rooms=("_nearest_room", lambda x: sorted(set(str(v) for v in x if pd.notna(v) and str(v) != "None"))),
        )
        .reset_index()
        .rename(columns={"_node": "node"})
    )

    return grouped, matched, unmatched


def build_graph(nodes, edges):
    node_col = get_col(nodes, ["node_id"])
    G = nx.Graph()

    for _, row in nodes.iterrows():
        node_id = str(row[node_col]).strip()
        G.add_node(node_id, **{c: row[c] for c in nodes.columns if c != node_col})

    from_col = get_col(edges, ["from_node_id", "from", "source"])
    to_col = get_col(edges, ["to_node_id", "to", "target"])
    type_col = get_col(edges, ["edge_type", "type"])
    length_col = get_col(edges, ["length", "distance"], required=False)
    edge_id_col = get_col(edges, ["edge_id"], required=False)

    for idx, row in edges.iterrows():
        u = str(row[from_col]).strip()
        v = str(row[to_col]).strip()
        edge_type = normalize_edge_type(row[type_col])

        length = 1.0
        if edge_type == "walk" and length_col is not None and pd.notna(row[length_col]):
            length_value = pd.to_numeric(row[length_col], errors="coerce")
            if pd.isna(length_value):
                raise ValueError(
                    f"walk edge의 length/distance 값이 숫자가 아닙니다. "
                    f"edge row={idx + 1}, from={u}, to={v}, value={row[length_col]}"
                )
            length = float(length_value)

        edge_id = row[edge_id_col] if edge_id_col is not None else f"E{idx + 1}"
        G.add_edge(u, v, edge_id=edge_id, edge_type=edge_type, length=length)

    return G


def has_elevator(G):
    return any(str(d.get("edge_type", "")).lower() == "elevator" for _, _, d in G.edges(data=True))


def edge_cost(edge_data, params=PARAMS):
    edge_type = str(edge_data.get("edge_type", "")).lower()
    if edge_type == "walk":
        return float(edge_data.get("length", 1.0)) * params["walk_cost_per_meter"]
    if edge_type == "stair":
        return float(params["stair_cost"])
    if edge_type == "elevator":
        return float(params["elevator_cost"])
    return float(edge_data.get("length", 1.0)) * params["walk_cost_per_meter"]


def decide_stair_policy(G, grouped, dispatch_people):
    elevator_exists = has_elevator(G)

    if not elevator_exists or grouped.empty:
        return {
            "elevator_exists": elevator_exists,
            "dispatch_people": dispatch_people,
            "avoid_stairs": False,
            "penalize_consecutive_stairs": True,
            "reason": "엘리베이터가 없는 건물이므로 계단 회피 정책 미적용" if not elevator_exists else "수거 대상 없음",
        }

    max_required = float(grouped["max_required_people"].max())
    has_item_required_ge_2 = max_required >= 2
    required_exceeds_dispatch = max_required > float(dispatch_people)
    avoid_stairs = has_item_required_ge_2 or required_exceeds_dispatch

    reasons = []
    if has_item_required_ge_2:
        reasons.append("필요인원수 2 이상 물품 존재")
    if required_exceeds_dispatch:
        reasons.append("필요인원수 최댓값이 투입인원수 초과")

    return {
        "elevator_exists": True,
        "dispatch_people": dispatch_people,
        "max_required_people": max_required,
        "avoid_stairs": avoid_stairs,
        "penalize_consecutive_stairs": True,
        "reason": ", ".join(reasons) if reasons else "계단 사용 가능, 연속 계단 사용은 가능한 회피",
    }


def add_metric(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def reconstruct_state_path(prev, state):
    states = [state]
    while state in prev:
        state = prev[state]
        states.append(state)
    states.reverse()
    return [node for node, _ in states]



def plain_dijkstra_same_floor_walk_only(G, source, target, allowed_floor, params=PARAMS):
    """
    같은 층 안에서만 이동하는 최단경로.
    층 이동 edge(stair/elevator)는 절대 사용하지 않는다.

    목적:
    - 출발 노드 -> 출발층 엘리베이터 노드
    - 도착층 엘리베이터 노드 -> 도착 노드
    구간을 계산할 때, 중간층 환승이 끼어드는 것을 원천 차단한다.
    """
    source = str(source)
    target = str(target)

    if source == target:
        return 0.0, [source]

    pq = [(0.0, source)]
    dist = {source: 0.0}
    prev = {}

    while pq:
        cost, u = heapq.heappop(pq)

        if cost != dist.get(u, float("inf")):
            continue

        if u == target:
            path = [target]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return cost, path

        for v, data in G[u].items():
            e_type = str(data.get("edge_type", "")).lower()

            # 같은 층 내부 보행 경로만 허용한다.
            if e_type in {"stair", "elevator"}:
                continue

            u_floor = get_node_floor_num(G, u)
            v_floor = get_node_floor_num(G, v)

            if u_floor != allowed_floor or v_floor != allowed_floor:
                continue

            next_cost = cost + edge_cost(data, params)

            if next_cost < dist.get(v, float("inf")):
                dist[v] = next_cost
                prev[v] = u
                heapq.heappush(pq, (next_cost, v))

    raise nx.NetworkXNoPath(
        f"{source}에서 {target}까지 {allowed_floor}층 내부 보행 경로가 없습니다."
    )


def elevator_only_path_between_floors(G, elev_from, target_floor, params=PARAMS):
    """
    특정 엘리베이터 노드에서 시작해서, elevator edge만 사용해 target_floor의
    엘리베이터 노드까지 가는 최단 경로를 찾는다.

    중요:
    - walk 금지
    - stair 금지
    - elevator edge만 허용
    - 따라서 중간층에서 내려서 복도를 걷고 다른 엘리베이터로 갈아타는 경로가 불가능하다.
    """
    elev_from = str(elev_from)

    pq = [(0.0, elev_from)]
    dist = {elev_from: 0.0}
    prev = {}

    while pq:
        cost, u = heapq.heappop(pq)

        if cost != dist.get(u, float("inf")):
            continue

        if u != elev_from and get_node_floor_num(G, u) == target_floor:
            path = [u]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return cost, path

        for v, data in G[u].items():
            e_type = str(data.get("edge_type", "")).lower()
            if e_type != "elevator":
                continue

            next_cost = cost + edge_cost(data, params)

            if next_cost < dist.get(v, float("inf")):
                dist[v] = next_cost
                prev[v] = u
                heapq.heappush(pq, (next_cost, v))

    raise nx.NetworkXNoPath(
        f"{elev_from}에서 {target_floor}층까지 elevator-only 경로가 없습니다."
    )


def find_forced_elevator_route_between_floors(G, source, target, params=PARAMS):
    """
    출발층과 도착층이 다를 때, 엘리베이터를 이용한 강제 경로를 만든다.

    경로 구조를 반드시 아래 형태로 제한한다.

    출발 노드
    -> 출발층 내부 보행
    -> 출발층 엘리베이터 노드
    -> elevator edge만 연속 사용
    -> 도착층 엘리베이터 노드
    -> 도착층 내부 보행
    -> 도착 노드

    이 함수가 성공하면, 5F -> 1F 이동 중 2F에서 내려서 복도를 걷고
    다른 엘리베이터로 갈아타는 경로는 절대 나올 수 없다.
    """
    source = str(source)
    target = str(target)

    source_floor = get_node_floor_num(G, source)
    target_floor = get_node_floor_num(G, target)

    if source_floor is None or target_floor is None or source_floor == target_floor:
        return None

    elevator_nodes_on_source_floor = [
        n for n, data in G.nodes(data=True)
        if get_node_floor_num(G, n) == source_floor
        and any(str(edge_data.get("edge_type", "")).lower() == "elevator" for _, edge_data in G[n].items())
    ]

    best_metric = (10**18, 10**18, float("inf"))
    best_path = None

    for elev_start in elevator_nodes_on_source_floor:
        try:
            walk_cost_1, path_to_elevator = plain_dijkstra_same_floor_walk_only(
                G, source, elev_start, source_floor, params
            )

            elevator_cost_value, elevator_path = elevator_only_path_between_floors(
                G, elev_start, target_floor, params
            )

            elev_end = elevator_path[-1]

            walk_cost_2, path_from_elevator = plain_dijkstra_same_floor_walk_only(
                G, elev_end, target, target_floor, params
            )
        except nx.NetworkXNoPath:
            continue

        # path_to_elevator[-1] == elevator_path[0]
        # elevator_path[-1] == path_from_elevator[0]
        full_path = (
            path_to_elevator
            + elevator_path[1:]
            + path_from_elevator[1:]
        )

        metric = (
            0,
            0,
            walk_cost_1 + elevator_cost_value + walk_cost_2,
        )

        if metric < best_metric:
            best_metric = metric
            best_path = full_path

    if best_path is None:
        return None

    return best_metric, best_path


def constrained_dijkstra(G, source, target, stair_policy, params=PARAMS):
    source = str(source)
    target = str(target)

    source_floor = get_node_floor_num(G, source)
    target_floor = get_node_floor_num(G, target)

    # 핵심 수정:
    # 층이 다르고 엘리베이터-only 경로가 가능하면, 일반 Dijkstra를 돌리지 않고
    # 엘리베이터 강제 경로를 먼저 반환한다.
    #
    # 이 방식은 단일 5F->1F elevator edge가 없어도 동작한다.
    # 예: 5F N108 -> 4F N72 -> 3F N50 -> 2F N32 -> 1F N10
    # 위처럼 같은 엘리베이터 축이 여러 elevator edge로 이어진 경우도
    # 하나의 직통 엘리베이터 이동처럼 처리한다.
    forced_elevator = find_forced_elevator_route_between_floors(
        G, source, target, params
    )

    if forced_elevator is not None:
        return forced_elevator

    # 엘리베이터 강제 경로가 불가능한 경우에만 기존 제약 Dijkstra를 사용한다.
    if source_floor is not None and target_floor is not None:
        min_allowed_floor = min(source_floor, target_floor)
        max_allowed_floor = max(source_floor, target_floor)
    else:
        min_allowed_floor = None
        max_allowed_floor = None

    start_metric = (0, 0, 0.0)
    pq = [(start_metric, source, "none")]
    dist = {(source, "none"): start_metric}
    prev = {}

    while pq:
        metric, u, last_edge_type = heapq.heappop(pq)
        state = (u, last_edge_type)

        if metric != dist.get(state, (10**18, 10**18, float("inf"))):
            continue

        if u == target:
            path = reconstruct_state_path(prev, state)
            return metric, path

        for v, data in G[u].items():
            e_type = str(data.get("edge_type", "")).lower()

            u_floor = get_node_floor_num(G, u)
            v_floor = get_node_floor_num(G, v)

            if min_allowed_floor is not None and max_allowed_floor is not None:
                if v_floor is not None and not (min_allowed_floor <= v_floor <= max_allowed_floor):
                    continue

            stair_count = 1 if (
                stair_policy.get("avoid_stairs")
                and e_type == "stair"
            ) else 0

            consecutive_stair_count = 1 if (
                stair_policy.get("penalize_consecutive_stairs")
                and last_edge_type == "stair"
                and e_type == "stair"
            ) else 0

            step_metric = (
                stair_count,
                consecutive_stair_count,
                edge_cost(data, params),
            )

            next_metric = add_metric(metric, step_metric)
            next_state = (v, e_type)

            if next_metric < dist.get(next_state, (10**18, 10**18, float("inf"))):
                dist[next_state] = next_metric
                prev[next_state] = state
                heapq.heappush(pq, (next_metric, v, e_type))

    raise nx.NetworkXNoPath(f"{source}에서 {target}까지 도달 가능한 경로가 없습니다.")

def precompute_pair_paths(G, nodes, stair_policy):
    pair_metric = {}
    pair_path = {}

    for a in nodes:
        for b in nodes:
            if a == b:
                pair_metric[(a, b)] = (0, 0, 0.0)
                pair_path[(a, b)] = [a]
            else:
                metric, path = constrained_dijkstra(G, a, b, stair_policy)
                pair_metric[(a, b)] = metric
                pair_path[(a, b)] = path

    return pair_metric, pair_path


def route_metric(route, pair_metric):
    total = (0, 0, 0.0)
    for i in range(len(route) - 1):
        total = add_metric(total, pair_metric[(route[i], route[i + 1])])
    return total


def exact_tsp(start, visit_nodes, pair_metric, return_to_start=True):
    targets = [n for n in visit_nodes if n != start]
    best_route = None
    best_metric = (10**18, 10**18, float("inf"))

    for perm in itertools.permutations(targets):
        route = [start, *perm]
        if return_to_start:
            route.append(start)

        metric = route_metric(route, pair_metric)
        if metric < best_metric:
            best_metric = metric
            best_route = route

    return best_route, best_metric


def nearest_neighbor(start, visit_nodes, pair_metric, return_to_start=True):
    current = start
    unvisited = set(visit_nodes)
    unvisited.discard(start)
    route = [start]

    while unvisited:
        next_node = min(unvisited, key=lambda n: pair_metric[(current, n)])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    if return_to_start:
        route.append(start)

    return route, route_metric(route, pair_metric)


def two_opt(route, pair_metric):
    if len(route) <= 4:
        return route, route_metric(route, pair_metric)

    best = route[:]
    best_metric = route_metric(best, pair_metric)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                candidate = best[:]
                candidate[i:j] = reversed(candidate[i:j])
                candidate_metric = route_metric(candidate, pair_metric)

                if candidate_metric < best_metric:
                    best = candidate
                    best_metric = candidate_metric
                    improved = True

    return best, best_metric


def solve_visit_order(G, visit_nodes, start_nodes, stair_policy, params=PARAMS):
    all_needed_nodes = sorted(set(visit_nodes) | set(start_nodes))
    pair_metric, pair_path = precompute_pair_paths(G, all_needed_nodes, stair_policy)

    best = None

    for start in start_nodes:
        n_requests = len(set(visit_nodes))

        if n_requests <= params["exact_tsp_threshold"]:
            algorithm = "Dijkstra + Exact TSP"
            route, metric = exact_tsp(start, visit_nodes, pair_metric, params["return_to_start"])
        else:
            algorithm = "Dijkstra + Nearest Neighbor + 2-opt"
            route, _ = nearest_neighbor(start, visit_nodes, pair_metric, params["return_to_start"])
            route, metric = two_opt(route, pair_metric)

        candidate = {
            "start_node": start,
            "route": route,
            "total_metric": metric,
            "total_stair_edges": metric[0],
            "total_consecutive_stair_edges": metric[1],
            "total_cost": metric[2],
            "algorithm": algorithm,
            "pair_path": pair_path,
        }

        if best is None or candidate["total_metric"] < best["total_metric"]:
            best = candidate

    return best


def edge_steps_from_node_path(G, path_nodes):
    steps = []

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        data = G[u][v]

        steps.append({
            "from": u,
            "to": v,
            "edge_id": data.get("edge_id"),
            "edge_type": data.get("edge_type"),
            "length": data.get("length", ""),
            "edge_cost": edge_cost(data),
        })

    return steps


def build_detailed_result(G, solution, node_info):
    route = solution["route"]
    pair_path = solution["pair_path"]
    detailed = []

    for idx in range(len(route) - 1):
        a = route[idx]
        b = route[idx + 1]
        path_nodes = pair_path[(a, b)]
        edge_steps = edge_steps_from_node_path(G, path_nodes)

        stair_count = sum(1 for e in edge_steps if str(e["edge_type"]).lower() == "stair")

        consecutive_stair_count = 0
        last_type = None
        for e in edge_steps:
            e_type = str(e["edge_type"]).lower()
            if last_type == "stair" and e_type == "stair":
                consecutive_stair_count += 1
            last_type = e_type

        detailed.append({
            "segment_no": idx + 1,
            "from_visit_node": a,
            "to_visit_node": b,
            "from_rooms": node_info.get(a, {}).get("rooms", []),
            "to_rooms": node_info.get(b, {}).get("rooms", []),
            "path_nodes": path_nodes,
            "path_edges": edge_steps,
            "segment_cost": sum(e["edge_cost"] for e in edge_steps),
            "segment_stair_edges": stair_count,
            "segment_consecutive_stair_edges": consecutive_stair_count,
        })

    return detailed


def build_full_node_path(solution):
    route = solution["route"]
    pair_path = solution["pair_path"]
    full_path = []

    for idx in range(len(route) - 1):
        a = route[idx]
        b = route[idx + 1]
        segment_path = pair_path[(a, b)]

        if idx == 0:
            full_path.extend(segment_path)
        else:
            full_path.extend(segment_path[1:])

    return full_path


def build_pickup_room_order(visit_route, node_info):
    pickup_room_order = []
    visited_pickup_nodes = set()

    for node in visit_route:
        if node in node_info and node not in visited_pickup_nodes:
            pickup_room_order.extend(map(str, node_info[node].get("rooms", [])))
            visited_pickup_nodes.add(node)

    return pickup_room_order


def build_pickup_node_order(visit_route, node_info):
    pickup_node_order = []
    visited_pickup_nodes = set()

    for node in visit_route:
        if node in node_info and node not in visited_pickup_nodes:
            pickup_node_order.append(node)
            visited_pickup_nodes.add(node)

    return pickup_node_order


def load_building_files(building, nodes_edges_dir):
    base = Path(nodes_edges_dir)
    node_path = base / f"{building}_nodes.csv"
    edge_path = base / f"{building}_edges.csv"

    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError(
            f"'{building}'의 node/edge 파일을 찾을 수 없습니다. "
            f"필요 파일: {node_path}, {edge_path}"
        )

    return read_csv_safely(node_path), read_csv_safely(edge_path), node_path, edge_path


def solve_building(building, building_items, dispatch_people, params=PARAMS):
    nodes, edges, node_path, edge_path = load_building_files(building, params["nodes_edges_dir"])

    grouped, matched_items, unmatched = map_items_to_nodes(building_items, nodes)

    if grouped.empty:
        return {
            "building": building,
            "status": "no_matched_nodes",
            "node_file": str(node_path),
            "edge_file": str(edge_path),
            "matched_items": matched_items,
            "unmatched": unmatched,
        }

    G = build_graph(nodes, edges)
    start_nodes = get_start_nodes(nodes)
    visit_nodes = grouped["node"].astype(str).tolist()

    stair_policy = decide_stair_policy(G, grouped, dispatch_people)
    solution = solve_visit_order(G, visit_nodes, start_nodes, stair_policy, params)

    node_info = {
        str(row["node"]): {
            "rooms": row["rooms"],
            "request_count": int(row["request_count"]),
            "total_quantity": float(row["total_quantity"]),
            "max_required_people": float(row["max_required_people"]),
            "input_row_ids": row["input_row_ids"],
            "mapping_types": row.get("mapping_types", []),
            "nearest_rooms": row.get("nearest_rooms", []),
        }
        for _, row in grouped.iterrows()
    }

    return {
        "building": building,
        "status": "ok",
        "node_file": str(node_path),
        "edge_file": str(edge_path),
        "start_node": solution["start_node"],
        "algorithm": solution["algorithm"],
        "visit_route": solution["route"],
        "full_node_path": build_full_node_path(solution),
        "pickup_room_order": build_pickup_room_order(solution["route"], node_info),
        "pickup_node_order": build_pickup_node_order(solution["route"], node_info),
        "total_cost": solution["total_cost"],
        "total_stair_edges": solution["total_stair_edges"],
        "total_consecutive_stair_edges": solution["total_consecutive_stair_edges"],
        "dispatch_people": dispatch_people,
        "stair_policy": stair_policy,
        "matched_items": matched_items,
        "node_info": node_info,
        "unmatched": unmatched,
        "detailed": build_detailed_result(G, solution, node_info),
    }


def build_node_lookup(result):
    nodes = read_csv_safely(result["node_file"])

    node_col = get_col(nodes, ["node_id"])
    x_col = get_col(nodes, ["x", "coord_x", "node_x"])
    y_col = get_col(nodes, ["y", "coord_y", "node_y"])
    floor_col = get_col(nodes, ["floor", "층", "floor_num"])
    assigned_col = get_col(nodes, ["assigned_rooms", "assigned_room", "room", "rooms", "대표호수", "호수"], required=False)
    type_col = get_col(nodes, ["node_type", "type", "종류", "노드유형"], required=False)

    lookup = {}

    for _, row in nodes.iterrows():
        node_id = str(row[node_col]).strip()

        node_type = ""
        if type_col is not None and pd.notna(row[type_col]):
            node_type = str(row[type_col]).strip()

        lookup[node_id] = {
            "id": node_id,
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "floor": normalize_floor(row[floor_col]),
            "assigned_rooms": split_assigned_rooms(row[assigned_col]) if assigned_col is not None else [],
            "node_type": node_type,
        }

    return lookup


def build_pickup_item_map(result):
    item_map = {}
    matched_items = result.get("matched_items")

    if matched_items is None or len(matched_items) == 0:
        return item_map

    for _, row in matched_items.iterrows():
        node = str(row["_node"])
        room = str(row["_room"])
        item_name = str(row.get("품명", "물품"))
        quantity = to_float(row.get("수량", 1), 1)

        item_map.setdefault(node, [])
        item_map[node].append({
            # 안내 문구에는 원래 신청 호수를 그대로 사용한다.
            "호수": room,
            "품명": item_name,
            "수량": quantity,
            "input_row_id": int(row["_input_row_id"]),
            "mapping_type": str(row.get("_mapping_type", "exact")),
            "nearest_room": None if pd.isna(row.get("_nearest_room")) else str(row.get("_nearest_room")),
            "nearest_room_distance": None if pd.isna(row.get("_nearest_room_distance")) else float(row.get("_nearest_room_distance")),
        })

    return item_map


def is_floor_transition_edge(edge, node_lookup):
    u = str(edge["from"])
    v = str(edge["to"])
    edge_type = str(edge.get("edge_type", "")).lower()

    if edge_type in {"stair", "elevator"}:
        return True

    if u in node_lookup and v in node_lookup:
        return node_lookup[u]["floor"] != node_lookup[v]["floor"]

    return False


def make_node_for_ui(node_id, node_lookup, result, pickup_item_map):
    node = node_lookup[node_id].copy()
    node_type = str(node.get("node_type", ""))
    node_type_lower = node_type.lower()

    node["is_start"] = node_id == result["start_node"]
    node["is_pickup"] = node_id in set(result["pickup_node_order"])
    node["is_route_node"] = node_id in set(result["full_node_path"])
    node["is_stair"] = "stair" in node_type_lower or "계단" in node_type
    node["is_elevator"] = (
        "elevator" in node_type_lower
        or "lift" in node_type_lower
        or "엘리베이터" in node_type
        or "엘베" in node_type
    )
    node["pickup_items"] = pickup_item_map.get(node_id, [])

    return node


def make_pickup_text(node_id, result, pickup_item_map):
    items = pickup_item_map.get(node_id, [])

    if not items:
        info = result["node_info"].get(node_id, {})
        rooms = info.get("rooms", [])
        if rooms:
            return f"{', '.join(map(str, rooms))}호에서 물품을 수거하세요."
        return "물품을 수거하세요."

    rooms = sorted(set(str(item["호수"]) for item in items))
    item_names = [str(item["품명"]) for item in items]

    return f"{', '.join(rooms)}호에서 {', '.join(item_names)}을 수거하세요."


def make_floor_transition_text(edge, node_lookup):
    v = str(edge["to"])
    to_floor = node_lookup[v]["floor"] if v in node_lookup else ""
    edge_type = str(edge.get("edge_type", "")).lower()

    if edge_type == "stair":
        move_type = "계단"
    elif edge_type == "elevator":
        move_type = "엘리베이터"
    else:
        move_type = "층 이동 경로"

    return f"{move_type}를 이용해서 {to_floor}로 이동하세요."


def make_floor_transition_pickup_text(edge, node_lookup, node_id, result, pickup_item_map):
    v = str(edge["to"])
    to_floor = node_lookup[v]["floor"] if v in node_lookup else ""
    edge_type = str(edge.get("edge_type", "")).lower()

    if edge_type == "stair":
        move_type = "계단"
    elif edge_type == "elevator":
        move_type = "엘리베이터"
    else:
        move_type = "층 이동 경로"

    pickup_text = make_pickup_text(node_id, result, pickup_item_map)

    return f"{move_type}를 이용해서 {to_floor}로 이동한 뒤, {pickup_text}"


def make_move_to_transition_text(edge):
    edge_type = str(edge.get("edge_type", "")).lower()

    if edge_type == "stair":
        return "계단으로 이동하세요."
    if edge_type == "elevator":
        return "엘리베이터로 이동하세요."
    return "층 이동 지점으로 이동하세요."


def make_route_edge_for_ui(edge, node_lookup, order):
    u = str(edge["from"])
    v = str(edge["to"])

    return {
        "order": order,
        "from": u,
        "to": v,
        "from_floor": node_lookup[u]["floor"] if u in node_lookup else None,
        "to_floor": node_lookup[v]["floor"] if v in node_lookup else None,
        "edge_id": edge.get("edge_id"),
        "edge_type": edge.get("edge_type"),
        "length": edge.get("length"),
        "edge_cost": edge.get("edge_cost"),
        "is_floor_transition": is_floor_transition_edge(edge, node_lookup),
    }


def build_ordered_route_edges(result, node_lookup):
    ordered_edges = []

    for seg in result["detailed"]:
        for edge in seg["path_edges"]:
            ordered_edges.append(
                make_route_edge_for_ui(edge, node_lookup, len(ordered_edges) + 1)
            )

    return ordered_edges


def build_navigation_steps(result):
    node_lookup = build_node_lookup(result)
    pickup_item_map = build_pickup_item_map(result)
    ordered_edges = build_ordered_route_edges(result, node_lookup)

    pickup_nodes = set(result["pickup_node_order"])
    visited_pickups = set()
    steps = []

    if not result["full_node_path"]:
        return steps

    current_nodes = [result["full_node_path"][0]]
    current_edges = []

    def flush_step(step_type, guide_text, trigger_node=None, transition_edges=None):
        nonlocal current_nodes, current_edges, steps

        unique_nodes = []
        seen = set()

        for n in current_nodes:
            if n in node_lookup and n not in seen:
                unique_nodes.append(n)
                seen.add(n)

        step_nodes = [
            make_node_for_ui(n, node_lookup, result, pickup_item_map)
            for n in unique_nodes
        ]

        step = {
            "step_no": len(steps) + 1,
            "step_type": step_type,
            "guide_text": guide_text,
            "floor": step_nodes[-1]["floor"] if step_nodes else None,
            "node_sequence": current_nodes[:],
            "nodes": step_nodes,
            "edges": current_edges[:],
        }

        if trigger_node is not None:
            step["trigger_node"] = trigger_node

        if transition_edges is not None:
            step["transition_edges"] = transition_edges
            if transition_edges:
                step["transition_edge"] = transition_edges[-1]

        steps.append(step)

        current_nodes = [current_nodes[-1]] if current_nodes else []
        current_edges = []

    def append_edge_to_current(edge):
        nonlocal current_nodes, current_edges
        u = edge["from"]
        v = edge["to"]

        if not current_nodes:
            current_nodes.append(u)

        if current_nodes[-1] != u:
            current_nodes.append(u)

        current_edges.append(edge)
        current_nodes.append(v)

    i = 0

    while i < len(ordered_edges):
        edge = ordered_edges[i]

        # 층 이동 edge를 만나면, 그 edge를 타기 전까지의 같은 층 이동을 먼저 분리
        if edge["is_floor_transition"]:
            u = edge["from"]

            # current_edges가 비어 있지 않다면, 이미 같은 층에서 층이동 노드까지 이동한 경로가 있음
            if current_edges:
                if current_nodes[-1] != u:
                    current_nodes.append(u)

                flush_step(
                    step_type="move_to_transition",
                    guide_text=make_move_to_transition_text(edge),
                    trigger_node=u,
                )

            # 이제 실제 층 이동 edge들만 하나의 step으로 묶음
            current_nodes = [u]
            current_edges = []

            transition_edges = []
            move_type = str(edge.get("edge_type", "")).lower()

            j = i
            while j < len(ordered_edges):
                next_edge = ordered_edges[j]

                if not next_edge["is_floor_transition"]:
                    break

                next_type = str(next_edge.get("edge_type", "")).lower()

                # 엘리베이터->엘리베이터, 계단->계단처럼 같은 이동수단일 때만 병합
                if next_type != move_type:
                    break

                append_edge_to_current(next_edge)
                transition_edges.append(next_edge)
                j += 1

            last_edge = transition_edges[-1]
            target_node = last_edge["to"]

            if target_node in pickup_nodes and target_node not in visited_pickups:
                visited_pickups.add(target_node)
                guide_text = make_floor_transition_pickup_text(
                    last_edge,
                    node_lookup,
                    target_node,
                    result,
                    pickup_item_map,
                )
                step_type = "floor_transition_pickup"
            else:
                guide_text = make_floor_transition_text(last_edge, node_lookup)
                step_type = "floor_transition"

            flush_step(
                step_type=step_type,
                guide_text=guide_text,
                trigger_node=target_node,
                transition_edges=transition_edges,
            )

            i = j
            continue

        # 일반 이동 edge
        append_edge_to_current(edge)
        v = edge["to"]

        # 일반 이동 중 수거 노드에 도착하면 pickup step으로 묶음
        if v in pickup_nodes and v not in visited_pickups:
            visited_pickups.add(v)
            flush_step(
                step_type="pickup",
                guide_text=make_pickup_text(v, result, pickup_item_map),
                trigger_node=v,
            )

        i += 1

    if len(current_nodes) >= 2 or current_edges:
        flush_step(
            step_type="exit",
            guide_text="출구로 이동하세요.",
            trigger_node=current_nodes[-1] if current_nodes else None,
        )
    elif not steps:
        current_nodes = [result["start_node"]]
        flush_step(
            step_type="exit",
            guide_text="출구로 이동하세요.",
            trigger_node=result["start_node"],
        )

    for idx, step in enumerate(steps):
        step["is_last_step"] = idx == len(steps) - 1

    return steps

def build_ui_visualization_dict(result):
    if result["status"] != "ok":
        out = {
            "건물명": result["building"],
            "상태": result["status"],
            "steps": [],
        }

        if "unmatched" in result and len(result["unmatched"]) > 0:
            out["매핑실패"] = (
                result["unmatched"]
                .drop(columns=[c for c in result["unmatched"].columns if c.startswith("_")], errors="ignore")
                .to_dict("records")
            )

        return out

    return {
        "건물명": result["building"],
        "상태": result["status"],
        "시작노드": result["start_node"],
        "총방문노드순서": result["full_node_path"],
        "수거노드순서": result["pickup_node_order"],
        "수거호수순서": result["pickup_room_order"],
        "steps": build_navigation_steps(result),
        "meta": {
            "algorithm": result["algorithm"],
            "total_cost": result["total_cost"],
            "dispatch_people": result["dispatch_people"],
            "total_stair_edges": result["total_stair_edges"],
            "total_consecutive_stair_edges": result["total_consecutive_stair_edges"],
            "stair_policy": result["stair_policy"],
            "nearest_room_mapping_applied": any(
                str(v.get("_mapping_type", "")) == "nearest_room"
                for v in result.get("matched_items", pd.DataFrame()).to_dict("records")
            ) if result.get("matched_items") is not None else False,
        },
    }


def save_outputs(results, output_dir, raw_input):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = {
        "입력": raw_input,
        "건물별경로안내": [
            build_ui_visualization_dict(r)
            for r in results
        ],
    }

    with open(output_dir / "building_routes_for_ui.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)


def print_result(result):
    print("\n" + "=" * 60)
    print(f"🏢 건물: {result['building']}")
    print("=" * 60)

    if result["status"] != "ok":
        print(f"상태: {result['status']}")
        return

    print(f"시작 노드: {result['start_node']}")
    print(f"투입인원수: {result['dispatch_people']}")
    print(f"알고리즘: {result['algorithm']}")
    print(f"총 비용: {result['total_cost']:.2f}")
    print(f"사용 계단 edge 수: {result['total_stair_edges']}")
    print(f"연속 계단 edge 수: {result['total_consecutive_stair_edges']}")

    print("\n✅ 총 방문 노드 순서")
    print(" -> ".join(result["full_node_path"]) if result["full_node_path"] else "이동 노드 없음")

    print("\n✅ 수거 노드")
    for node in result["pickup_node_order"]:
        info = result["node_info"][node]
        nearest_note = ""
        if "nearest_room" in info.get("mapping_types", []):
            nearest_note = f" | 대체매핑기준호수={info.get('nearest_rooms', [])}"

        print(
            f"- 수거노드={node} | 안내호수={info['rooms']} | "
            f"요청건수={info['request_count']} | 수량합={info['total_quantity']} | "
            f"최대필요인원={info['max_required_people']}"
            f"{nearest_note}"
        )


def main(params=PARAMS):
    schedule, dispatch_people, meta = load_json_schedule(params["input_json_path"], params)

    results = []

    for building, building_items in schedule.groupby("_building"):
        try:
            result = solve_building(building, building_items, dispatch_people, params)
        except Exception as e:
            result = {
                "building": building,
                "status": "error",
                "error": str(e),
                "unmatched": building_items.copy(),
            }
            print(f"\n❌ {building} 처리 중 오류: {e}")

        results.append(result)
        print_result(result)

    save_outputs(results, params["output_dir"], meta["raw_input"])

    print("\n" + "=" * 60)
    print(f"📁 결과 저장 완료: {params['output_dir']}")
    print(" - building_routes_for_ui.json")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ============================================================
# API 진입점 (api.py → run_optimizer 호출용)
# ============================================================
def run_optimizer(df, _df_avail=None, dispatch_people=None):
    """
    api.py에서 호출하는 진입점.
    df 컬럼: 신청번호, 품명, 설치장소, 수량, 필요인원수
    dispatch_people: 투입인원수 (None이면 필요인원수 최댓값으로 결정)
    반환: 건물별 수거 경로 및 내비게이션 steps 리스트
    """
    if dispatch_people is None:
        dispatch_people = float(df["필요인원수"].max()) if "필요인원수" in df.columns else PARAMS["default_dispatch_people"]

    schedule = df.copy().reset_index(drop=True)
    schedule["_input_row_id"] = schedule.index + 1

    parsed = schedule["설치장소"].apply(split_location_to_building_room)
    schedule["_building"]        = [p[0] for p in parsed]
    schedule["_room"]            = [p[1] for p in parsed]
    schedule["_quantity"]        = schedule["수량"].apply(lambda x: to_float(x, 1))
    schedule["_required_people"] = schedule["필요인원수"].apply(lambda x: to_float(x, 1))
    schedule["_dispatch_people"] = dispatch_people

    results = []
    for building, building_items in schedule.groupby("_building"):
        try:
            result = solve_building(building, building_items.reset_index(drop=True), dispatch_people, PARAMS)
        except Exception as e:
            print(f"[오류] {building} 처리 실패: {e}")
            results.append({"건물명": building, "상태": "error", "오류": str(e)})
            continue

        results.append(build_ui_visualization_dict(result))

    return results
