# indoor_route.py

import itertools
import heapq
import json
import re
from pathlib import Path
import pandas as pd

import networkx as nx


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

    # 입력 JSON의 건물명을 nodes/edges CSV 파일명에 사용하는 표준 건물명으로 변환한다.
    # 딕셔너리에 없는 장소명은 수거 대상에서 제외한다.
    "building_name_map": {
        # 입력 JSON 건물명 -> 코드 내부 건물명(CSV 파일명 기준)
        "애지원": "애지원",
        "우정원": "우정원",
        "공학관": "공학관",
        "공학실험동": "공학실험동",
        "체육대학관": "체육대학관",
        "외국어대학관": "외국어대학관",
        "멀티미디어교육관": "멀티미디어관",
        "멀티미디어관": "멀티미디어관",
        "글로벌관": "글로벌관",
        "멀티미디어교육관글로벌관": "글로벌관",
        "도예관": "도예관",
        "원예생명공학온실": "원예생명공학온실",
        "선승관": "선승관",
        "생명과학대학관": "생명과학대학관",
        "실험연구동A": "실험연구동A",
        "실험연구동B": "실험연구동B",
        "예술디자인대학관": "예디대",
        "예디대": "예디대",
        "국제경영대학관": "국제경영대학관",
        "학생회관": "학생회관",
        "중앙도서관": "중앙도서관",
        "도서관": "중앙도서관",
        "전자정보/응용과학대학관": "전정대",
        "전자정보응용과학대학관": "전정대",
        "전자정보·응용과학대학관": "전정대",
        "전정대": "전정대",
        "국제학관": "국제학관",
        "천문대": "천문대",
    },

    # 입력 가능 목록에는 있지만 현재 노드/엣지 데이터가 없어 처리하지 않는 건물명.
    "excluded_building_names": [
        "원자로센터",
        "제2기숙사(남)",
        "제2기숙사(여)",
        "한방재료가공",
    ],

    # 요청 호수가 nodes CSV의 assigned_rooms에 없을 때 사용할 건물별 행정실 호수.
    # 새 건물을 지원하려면 아래 딕셔너리에 "건물명": "행정실 호수"를 추가하면 된다.
    "admin_office_rooms": {
        "전정대": "203",
        "예디대": "214",
        "애지원": "202",
    },
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


def normalize_room_key(value):
    """
    호수 비교용 키를 만든다.
    예: 202, 202호, 202.0 -> 모두 '202'
    """
    if value is None or pd.isna(value):
        return ""

    text = normalize_text(value).upper()
    text = re.sub(r"호$", "", text)

    # CSV에서 숫자 호수가 202.0처럼 읽힌 경우 정리한다.
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]

    return text


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


def normalize_building_key(value):
    """건물명 비교용 키. 공백과 /, · 같은 구분 기호 차이를 무시한다."""
    text = normalize_text(value)
    return re.sub(r"[\/·ㆍ]", "", text)


def split_location_to_building_room(location, params=PARAMS):
    """
    설치장소의 시작 부분을 건물명 매핑 리스트와 비교해 표준 건물명과 호수를 분리한다.
    건물명 안에 공백이 있어도 처리하며, 매핑되지 않은 장소는 ("", None)을 반환한다.
    """
    if location is None or pd.isna(location):
        return "", None

    # 뒤쪽 괄호 설명(부서명 등)을 먼저 제거한다.
    # 예: '도서관301호 (산학협력단/…)' -> '도서관301호'
    cleaned = re.sub(r"\s*\(.*\)\s*$", "", str(location)).strip()

    location_key = normalize_building_key(cleaned)
    if not location_key:
        return "", None

    # 지원하지 않거나 하나의 건물로 확정할 수 없는 입력은 먼저 제외한다.
    excluded_keys = sorted(
        {
            normalize_building_key(name)
            for name in params.get("excluded_building_names", [])
        },
        key=len,
        reverse=True,
    )
    for excluded_key in excluded_keys:
        if location_key.startswith(excluded_key):
            return "", None

    alias_map = params.get("building_name_map", {})
    normalized_map = {
        normalize_building_key(alias): canonical
        for alias, canonical in alias_map.items()
    }

    # 짧은 건물명이 긴 건물명을 먼저 가로채지 않도록 긴 이름부터 비교한다.
    for alias_key in sorted(normalized_map, key=len, reverse=True):
        if location_key.startswith(alias_key):
            room = location_key[len(alias_key):] or None
            return normalized_map[alias_key], room

    return "", None


def load_json_schedule(input_json_path, params=PARAMS):
    data = read_json_safely(input_json_path)
    dispatch_people = to_float(data.get("투입인원수"), params["default_dispatch_people"])
    requests = data.get("신청서", [])

    if not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("JSON 입력에 '신청서' 리스트가 없거나 비어 있습니다.")

    rows = []
    excluded_requests = []

    for idx, item in enumerate(requests, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"신청서의 {idx}번째 항목이 객체가 아닙니다: {item}")

        building, room = split_location_to_building_room(
            item.get("설치장소", ""),
            params,
        )

        # 매핑 리스트에 없는 장소는 경로 계산 대상에서 제외한다.
        if not building:
            excluded = dict(item)
            excluded["_input_row_id"] = idx
            excluded_requests.append(excluded)
            continue

        row = dict(item)
        row["_input_row_id"] = idx
        row["_building"] = building
        row["_room"] = room
        row["_quantity"] = to_float(item.get("수량", 1), 1)
        row["_required_people"] = to_float(item.get("필요인원수", 1), 1)
        row["_dispatch_people"] = dispatch_people
        rows.append(row)

    schedule = pd.DataFrame(rows)

    if schedule.empty:
        schedule = pd.DataFrame(columns=[
            "_input_row_id",
            "_building",
            "_room",
            "_quantity",
            "_required_people",
            "_dispatch_people",
        ])

    return schedule, dispatch_people, {
        "raw_input": data,
        "excluded_requests": excluded_requests,
    }


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
            mapping[normalize_room_key(room)] = node_id
    return mapping




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


def map_items_to_nodes(building, building_items, nodes, params=PARAMS):
    """신청 호수를 노드에 매핑하고, 미등록 호수는 해당 건물 행정실로 보낸다."""
    room_to_node = build_room_to_node(nodes)
    items = building_items.copy()

    items["_original_room"] = items["_room"]
    items["_room_key"] = items["_room"].apply(normalize_room_key)
    items["_node"] = items["_room_key"].map(room_to_node)
    items["_mapping_type"] = items["_node"].apply(
        lambda node: "exact" if pd.notna(node) else None
    )
    items["_mapped_room"] = items["_room"]

    missing_mask = items["_node"].isna()
    if missing_mask.any():
        admin_room = params.get("admin_office_rooms", {}).get(building)
        if admin_room is None:
            raise ValueError(
                f"'{building}'의 행정실 호수가 PARAMS['admin_office_rooms']에 없습니다."
            )

        admin_room_key = normalize_room_key(admin_room)
        admin_node = room_to_node.get(admin_room_key)
        if admin_node is None:
            raise ValueError(
                f"'{building}' nodes 파일의 assigned_rooms에 "
                f"행정실 호수 '{admin_room}'가 없습니다."
            )

        items.loc[missing_mask, "_room"] = str(admin_room)
        items.loc[missing_mask, "_room_key"] = admin_room_key
        items.loc[missing_mask, "_node"] = admin_node
        items.loc[missing_mask, "_mapping_type"] = "admin_office"
        items.loc[missing_mask, "_mapped_room"] = str(admin_room)

    unmatched = items[items["_node"].isna()].copy()
    matched = items.dropna(subset=["_node"]).copy()

    grouped = (
        matched.groupby("_node")
        .agg(
            rooms=("_room", lambda values: sorted(set(str(v) for v in values if pd.notna(v)))),
            request_count=("_room", "size"),
            total_quantity=("_quantity", "sum"),
            max_required_people=("_required_people", "max"),
            input_row_ids=("_input_row_id", list),
            mapping_types=("_mapping_type", lambda values: sorted(set(str(v) for v in values if pd.notna(v)))),
            original_rooms=("_original_room", lambda values: sorted(set(str(v) for v in values if pd.notna(v)))),
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

    grouped, matched_items, unmatched = map_items_to_nodes(building, building_items, nodes, params)

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
            "nearest_rooms": [],
            "original_rooms": row.get("original_rooms", []),
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
        mapped_room = str(row.get("_mapped_room", row.get("_room", "")))
        original_room = str(row.get("_original_room", row.get("_room", "")))
        item_name = str(row.get("품명", "물품"))
        quantity = to_float(row.get("수량", 1), 1)

        item_map.setdefault(node, [])
        item_map[node].append({
            # 실제 이동 목적지 호수와 사용자가 요청한 원래 호수를 모두 보존한다.
            "호수": mapped_room,
            "요청호수": original_room,
            "품명": item_name,
            "수량": quantity,
            "input_row_id": int(row["_input_row_id"]),
            "mapping_type": str(row.get("_mapping_type", "exact")),
            "nearest_room": None,
            "nearest_room_distance": None,
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


def format_room_label(room):
    room_text = str(room).strip()
    if not room_text:
        return ""
    return room_text if room_text.endswith("호") else f"{room_text}호"


def format_quantity(quantity):
    quantity = float(quantity)
    if quantity.is_integer():
        return str(int(quantity))
    return f"{quantity:g}"


def make_admin_redirect_text(node_id, result, pickup_item_map):
    admin_items = [
        item for item in pickup_item_map.get(node_id, [])
        if str(item.get("mapping_type", "")) == "admin_office"
    ]

    if not admin_items:
        return ""

    original_rooms = sorted({
        format_room_label(item.get("요청호수", ""))
        for item in admin_items
        if str(item.get("요청호수", "")).strip()
    })

    if not original_rooms:
        return "요청하신 위치를 확인할 수 없어 행정실로 안내합니다."

    return (
        f"요청하신 {', '.join(original_rooms)}의 위치를 확인할 수 없어 "
        "행정실로 안내합니다."
    )


def make_pickup_text(node_id, result, pickup_item_map):
    items = pickup_item_map.get(node_id, [])

    if not items:
        info = result["node_info"].get(node_id, {})
        rooms = [format_room_label(room) for room in info.get("rooms", [])]
        if rooms:
            return f"{', '.join(rooms)}에서 물품을 수거하세요"
        return "물품을 수거하세요"

    has_admin_mapping = any(
        str(item.get("mapping_type", "")) == "admin_office"
        for item in items
    )

    if has_admin_mapping:
        location_text = "행정실"
    else:
        rooms = sorted({format_room_label(item["호수"]) for item in items})
        location_text = ", ".join(rooms)

    # 같은 품명은 한 번만 표시하고 수량을 합산한다.
    quantity_by_item = {}
    for item in items:
        item_name = str(item.get("품명", "물품")).strip() or "물품"
        quantity_by_item[item_name] = (
            quantity_by_item.get(item_name, 0.0)
            + to_float(item.get("수량", 1), 1)
        )

    item_text = ", ".join(
        f"{item_name} {format_quantity(quantity)}개"
        for item_name, quantity in quantity_by_item.items()
    )

    return f"{location_text}에서 {item_text}를 수거하세요"


def make_move_to_pickup_text(node_id, result, pickup_item_map):
    """수거지까지의 같은 층 이동 안내 문구. 예: '103호로 이동하세요'."""
    items = pickup_item_map.get(node_id, [])
    has_admin_mapping = any(
        str(item.get("mapping_type", "")) == "admin_office" for item in items
    )
    if has_admin_mapping:
        location_text = "행정실"
    elif items:
        rooms = sorted({format_room_label(item["호수"]) for item in items})
        location_text = ", ".join(rooms)
    else:
        info = result["node_info"].get(node_id, {})
        rooms = [format_room_label(room) for room in info.get("rooms", [])]
        location_text = ", ".join(rooms) if rooms else "수거지"
    return f"{location_text}로 이동하세요"


def floor_display(floor_str):
    """가이드 문구용 층 표시. 지하는 B 표기로: 0F->B1F, -1F->B2F. 1F 이상은 그대로."""
    s = str(floor_str).strip().upper()
    m = re.fullmatch(r"(-?\d+)F", s)
    if m:
        n = int(m.group(1))
        if n <= 0:
            return f"B{1 - n}F"
    return s


def make_floor_transition_text(edge, node_lookup):
    v = str(edge["to"])
    to_floor = floor_display(node_lookup[v]["floor"]) if v in node_lookup else ""
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
    to_floor = floor_display(node_lookup[v]["floor"]) if v in node_lookup else ""
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
        for edge_index, edge in enumerate(seg["path_edges"]):
            ui_edge = make_route_edge_for_ui(
                edge,
                node_lookup,
                len(ordered_edges) + 1,
            )
            ui_edge["segment_no"] = seg["segment_no"]
            ui_edge["segment_to_visit_node"] = seg["to_visit_node"]
            ui_edge["is_segment_start"] = edge_index == 0
            ordered_edges.append(ui_edge)

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
    pending_redirect_notice = ""

    def flush_step(step_type, guide_text, trigger_node=None, transition_edges=None):
        nonlocal current_nodes, current_edges, steps, pending_redirect_notice

        if pending_redirect_notice:
            guide_text = f"{pending_redirect_notice} {guide_text}"
            pending_redirect_notice = ""

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

        # 행정실 대체 노드로 향하는 새 구간이 시작되면,
        # 다음으로 출력되는 이동 안내 앞에 대체 안내 문구를 붙인다.
        if edge.get("is_segment_start"):
            destination_node = str(edge.get("segment_to_visit_node", ""))
            pending_redirect_notice = make_admin_redirect_text(
                destination_node,
                result,
                pickup_item_map,
            )

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

        # 일반 이동 중 수거 노드에 도착하면
        if v in pickup_nodes and v not in visited_pickups:
            visited_pickups.add(v)
            # ① 수거지까지의 '이동'을 별도 스텝으로 분리 → 프론트가 그 구간 지도를 그린다
            flush_step(
                step_type="move_to_pickup",
                guide_text=make_move_to_pickup_text(v, result, pickup_item_map),
                trigger_node=v,
            )
            # ② '수거' 스텝은 수거지 노드만 (이동 없음)
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
                str(v.get("_mapping_type", "")) == "admin_office"
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
        admin_note = " | 행정실 대체매핑" if "admin_office" in info.get("mapping_types", []) else ""

        print(
            f"- 수거노드={node} | 안내호수={info['rooms']} | "
            f"요청건수={info['request_count']} | 수량합={info['total_quantity']} | "
            f"최대필요인원={info['max_required_people']}"
            f"{admin_note}"
        )


def run_optimizer(df, _df_avail=None, dispatch_people=None, params=PARAMS):
    """
    api.py 진입점. DataFrame 을 받아 건물별 실내 수거 동선을 계산한다.
    df 컬럼: 신청번호, 품명, 설치장소, 수량, 필요인원수
    dispatch_people: 투입인원수 (None이면 필요인원수 최댓값)
    반환: 건물별 UI 시각화 dict 리스트 (build_ui_visualization_dict 형식)

    load_json_schedule 의 DataFrame 판(版)이다. 건물명 매핑·행정실 폴백 등
    모든 로직을 동일하게 공유하되, 입력만 파일 JSON 대신 DataFrame 을 받는다.
    """
    if dispatch_people is None:
        if "필요인원수" in df.columns and len(df) > 0:
            dispatch_people = float(pd.to_numeric(df["필요인원수"], errors="coerce").max())
        else:
            dispatch_people = float(params["default_dispatch_people"])

    schedule = df.copy().reset_index(drop=True)
    schedule["_input_row_id"] = schedule.index + 1

    parsed = schedule["설치장소"].apply(lambda loc: split_location_to_building_room(loc, params))
    schedule["_building"]        = [p[0] for p in parsed]
    schedule["_room"]            = [p[1] for p in parsed]
    schedule["_quantity"]        = schedule["수량"].apply(lambda x: to_float(x, 1))
    schedule["_required_people"] = schedule["필요인원수"].apply(lambda x: to_float(x, 1))
    schedule["_dispatch_people"] = dispatch_people

    results = []
    # 건물명 매핑 실패(building_name_map 미등록)는 처리 대상에서 제외한다.
    valid = schedule[schedule["_building"].astype(str) != ""]
    for building, building_items in valid.groupby("_building", sort=False):
        try:
            result = solve_building(building, building_items.reset_index(drop=True), dispatch_people, params)
            results.append(build_ui_visualization_dict(result))
        except Exception as e:
            print(f"[오류] {building} 처리 실패: {e}")
            results.append({"건물명": building, "상태": "error", "오류": str(e)})

    return results


def main(params=PARAMS):
    schedule, dispatch_people, meta = load_json_schedule(params["input_json_path"], params)

    excluded_requests = meta.get("excluded_requests", [])
    if excluded_requests:
        excluded_locations = [
            str(item.get("설치장소", ""))
            for item in excluded_requests
        ]
        print(
            f"\n⚠️ 건물명 매핑 실패로 {len(excluded_requests)}건 제외: "
            + ", ".join(excluded_locations)
        )

    results = []

    # 같은 건물의 요청을 한 묶음으로 최적화한 뒤 다음 건물을 처리한다.
    # sort=False로 JSON 입력에 처음 등장한 건물 순서를 유지한다.
    for building, building_items in schedule.groupby("_building", sort=False):
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
