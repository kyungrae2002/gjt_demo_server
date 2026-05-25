import itertools
import heapq
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# 파라미터
# ============================================================
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


# ============================================================
# 기본 유틸
# ============================================================
def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def read_json_safely(path: str | Path) -> dict:
    path = Path(path)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r") as f:
        return json.load(f)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def get_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"필수 컬럼이 없습니다. 후보={candidates}, 실제 컬럼={list(df.columns)}")
    return None


def to_float(value, default: float) -> float:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return float(default)
    return float(converted)


def normalize_edge_type(value) -> str:
    text = str(value).strip().lower()
    if text in {"walk", "corridor", "hallway", "복도", "통로", "보행"}:
        return "walk"
    if text in {"stair", "stairs", "계단"}:
        return "stair"
    if text in {"elevator", "lift", "엘리베이터", "엘베"}:
        return "elevator"
    return text if text else "other"


# ============================================================
# JSON 입력 전처리
# ============================================================
def split_location_to_building_room(location: str) -> tuple[str, str | None]:
    if location is None or pd.isna(location):
        return "", None

    text = str(location).strip()
    if not text:
        return "", None

    parts = text.split(maxsplit=1)
    building = parts[0].strip()
    room = parts[1].strip() if len(parts) > 1 else None
    return building, room


def load_json_schedule(input_json_path: str | Path, params: dict = PARAMS) -> tuple[pd.DataFrame, float, dict]:
    data = read_json_safely(input_json_path)

    dispatch_people = to_float(data.get("투입인원수"), params["default_dispatch_people"])
    requests = data.get("신청서", [])

    if not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("JSON 입력에 '신청서' 리스트가 없거나 비어 있습니다.")

    rows = []
    for idx, item in enumerate(requests, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"신청서의 {idx}번째 항목이 객체가 아닙니다: {item}")

        location = item.get("설치장소", "")
        building, room = split_location_to_building_room(location)

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
        bad = schedule[schedule["_building"].eq("")][["_input_row_id", "설치장소"]]
        raise ValueError(f"설치장소에서 건물명을 추출하지 못한 항목이 있습니다:\n{bad}")

    meta = {
        "dispatch_people": dispatch_people,
        "raw_input": data,
        "original_request_columns": [c for c in schedule.columns if not c.startswith("_")],
    }
    return schedule, dispatch_people, meta


# ============================================================
# node-room 매핑
# ============================================================
def split_assigned_rooms(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    parts = re.split(r"[;,/|]", text)
    return [p.strip() for p in parts if p.strip()]


def build_room_to_node(nodes: pd.DataFrame) -> dict[str, str]:
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


def get_start_nodes(nodes: pd.DataFrame) -> list[str]:
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


def map_items_to_nodes(building_items: pd.DataFrame, nodes: pd.DataFrame):
    room_to_node = build_room_to_node(nodes)
    items = building_items.copy()

    items["_room_key"] = items["_room"].apply(normalize_text)
    items["_node"] = items["_room_key"].map(room_to_node)

    unmatched = items[items["_node"].isna()].copy()
    matched = items.dropna(subset=["_node"]).copy()

    grouped = (
        matched.groupby("_node")
        .agg(
            rooms=("_room", lambda x: sorted(set(str(v) for v in x if pd.notna(v)))),
            request_count=("_room", "size"),
            total_quantity=("_quantity", "sum"),
            max_required_people=("_required_people", "max"),
            item_required_people=("_required_people", list),
            input_row_ids=("_input_row_id", list),
        )
        .reset_index()
        .rename(columns={"_node": "node"})
    )

    return grouped, matched, unmatched


# ============================================================
# 그래프 구성 및 edge 비용
# ============================================================
def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    node_col = get_col(nodes, ["node_id"])
    G = nx.Graph()

    for _, row in nodes.iterrows():
        node_id = str(row[node_col]).strip()
        G.add_node(
            node_id,
            **{c: row[c] for c in nodes.columns if c != node_col}
        )

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

        G.add_edge(
            u,
            v,
            edge_id=edge_id,
            edge_type=edge_type,
            length=length,
        )

    return G


def has_elevator(G: nx.Graph) -> bool:
    return any(str(d.get("edge_type", "")).lower() == "elevator" for _, _, d in G.edges(data=True))


def edge_cost(edge_data: dict, params: dict = PARAMS) -> float:
    edge_type = str(edge_data.get("edge_type", "")).lower()
    if edge_type == "walk":
        return float(edge_data.get("length", 1.0)) * params["walk_cost_per_meter"]
    if edge_type == "stair":
        return float(params["stair_cost"])
    if edge_type == "elevator":
        return float(params["elevator_cost"])
    return float(edge_data.get("length", 1.0)) * params["walk_cost_per_meter"]


def decide_stair_policy(G: nx.Graph, grouped: pd.DataFrame, dispatch_people: float) -> dict:
    elevator_exists = has_elevator(G)
    if not elevator_exists or grouped.empty:
        return {
            "elevator_exists": elevator_exists,
            "dispatch_people": dispatch_people,
            "avoid_stairs": False,
            "avoid_consecutive_elevators": False,
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
        "avoid_consecutive_elevators": True,
        "reason": ", ".join(reasons) if reasons else "계단 사용 가능, 연속 엘리베이터 사용은 가능한 회피",
    }


# ============================================================
# 제약 포함 최단경로
# ============================================================
def add_metric(a: tuple[int, int, float], b: tuple[int, int, float]) -> tuple[int, int, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def constrained_dijkstra(
    G: nx.Graph,
    source: str,
    target: str,
    stair_policy: dict,
    params: dict = PARAMS,
):
    source = str(source)
    target = str(target)

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

            stair_count = 1 if (stair_policy.get("avoid_stairs") and e_type == "stair") else 0
            consecutive_elevator_count = 1 if (
                stair_policy.get("avoid_consecutive_elevators")
                and last_edge_type == "elevator"
                and e_type == "elevator"
            ) else 0
            base_cost = edge_cost(data, params)

            step_metric = (stair_count, consecutive_elevator_count, base_cost)
            next_metric = add_metric(metric, step_metric)
            next_state = (v, e_type)

            if next_metric < dist.get(next_state, (10**18, 10**18, float("inf"))):
                dist[next_state] = next_metric
                prev[next_state] = state
                heapq.heappush(pq, (next_metric, v, e_type))

    raise nx.NetworkXNoPath(f"{source}에서 {target}까지 도달 가능한 경로가 없습니다.")


def reconstruct_state_path(prev: dict, state: tuple[str, str]) -> list[str]:
    states = [state]
    while state in prev:
        state = prev[state]
        states.append(state)
    states.reverse()
    return [node for node, _ in states]


def precompute_pair_paths(G: nx.Graph, nodes: list[str], stair_policy: dict):
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


# ============================================================
# TSP / NN / 2-opt
# ============================================================
def route_metric(route: list[str], pair_metric: dict) -> tuple[int, int, float]:
    total = (0, 0, 0.0)
    for i in range(len(route) - 1):
        total = add_metric(total, pair_metric[(route[i], route[i + 1])])
    return total


def exact_tsp(start: str, visit_nodes: list[str], pair_metric: dict, return_to_start: bool = True):
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


def nearest_neighbor(start: str, visit_nodes: list[str], pair_metric: dict, return_to_start: bool = True):
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


def two_opt(route: list[str], pair_metric: dict):
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


def solve_visit_order(G: nx.Graph, visit_nodes: list[str], start_nodes: list[str], stair_policy: dict, params: dict = PARAMS):
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
            "total_consecutive_elevators": metric[1],
            "total_cost": metric[2],
            "algorithm": algorithm,
            "pair_path": pair_path,
            "pair_metric": pair_metric,
        }

        if best is None or candidate["total_metric"] < best["total_metric"]:
            best = candidate

    return best


# ============================================================
# 상세 경로 복원
# ============================================================
def edge_steps_from_node_path(G: nx.Graph, path_nodes: list[str]) -> list[dict]:
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


def build_detailed_result(G: nx.Graph, solution: dict, node_info: dict[str, dict]):
    route = solution["route"]
    pair_path = solution["pair_path"]
    detailed = []

    for idx in range(len(route) - 1):
        a = route[idx]
        b = route[idx + 1]
        path_nodes = pair_path[(a, b)]
        edge_steps = edge_steps_from_node_path(G, path_nodes)

        stair_count = sum(1 for e in edge_steps if str(e["edge_type"]).lower() == "stair")
        consecutive_elevator_count = 0
        last_type = None
        for e in edge_steps:
            e_type = str(e["edge_type"]).lower()
            if last_type == "elevator" and e_type == "elevator":
                consecutive_elevator_count += 1
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
            "segment_consecutive_elevators": consecutive_elevator_count,
        })

    return detailed


def build_full_node_path(solution: dict) -> list[str]:
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


def build_pickup_room_order(visit_route: list[str], node_info: dict[str, dict]) -> list[str]:
    pickup_room_order = []
    visited_pickup_nodes = set()

    for node in visit_route:
        if node in node_info and node not in visited_pickup_nodes:
            pickup_room_order.extend(map(str, node_info[node].get("rooms", [])))
            visited_pickup_nodes.add(node)

    return pickup_room_order


def build_pickup_node_order(visit_route: list[str], node_info: dict[str, dict]) -> list[str]:
    pickup_node_order = []
    visited_pickup_nodes = set()

    for node in visit_route:
        if node in node_info and node not in visited_pickup_nodes:
            pickup_node_order.append(node)
            visited_pickup_nodes.add(node)

    return pickup_node_order


# ============================================================
# 건물 단위 처리
# ============================================================
def load_building_files(building: str, nodes_edges_dir: str | Path):
    base = Path(nodes_edges_dir)
    node_path = base / f"{building}_nodes.csv"
    edge_path = base / f"{building}_edges.csv"

    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError(
            f"'{building}'의 node/edge 파일을 찾을 수 없습니다. "
            f"필요 파일: {node_path}, {edge_path}"
        )

    return read_csv_safely(node_path), read_csv_safely(edge_path), node_path, edge_path


def solve_building(building: str, building_items: pd.DataFrame, dispatch_people: float, params: dict = PARAMS):
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
        }
        for _, row in grouped.iterrows()
    }

    detailed = build_detailed_result(G, solution, node_info)
    full_node_path = build_full_node_path(solution)
    pickup_room_order = build_pickup_room_order(solution["route"], node_info)
    pickup_node_order = build_pickup_node_order(solution["route"], node_info)

    return {
        "building": building,
        "status": "ok",
        "node_file": str(node_path),
        "edge_file": str(edge_path),
        "start_node": solution["start_node"],
        "algorithm": solution["algorithm"],
        "visit_route": solution["route"],
        "pickup_room_order": pickup_room_order,
        "pickup_node_order": pickup_node_order,
        "full_node_path": full_node_path,
        "total_cost": solution["total_cost"],
        "total_stair_edges": solution["total_stair_edges"],
        "total_consecutive_elevators": solution["total_consecutive_elevators"],
        "dispatch_people": dispatch_people,
        "stair_policy": stair_policy,
        "grouped": grouped,
        "matched_items": matched_items,
        "node_info": node_info,
        "unmatched": unmatched,
        "detailed": detailed,
    }


# ============================================================
# 출력 데이터 생성
# ============================================================
def result_to_public_dict(result: dict) -> dict:
    if result["status"] != "ok":
        out = {
            "건물명": result["building"],
            "상태": result["status"],
            "총방문노드순서": [],
            "수거노드": [],
        }
        if "unmatched" in result and len(result["unmatched"]) > 0:
            out["매핑실패"] = result["unmatched"].drop(columns=[c for c in result["unmatched"].columns if c.startswith("_")], errors="ignore").to_dict("records")
        return out

    pickup_nodes = []
    for visit_seq, node in enumerate(result["pickup_node_order"], start=1):
        info = result["node_info"].get(node, {})
        pickup_nodes.append({
            "방문순서": visit_seq,
            "노드": node,
            "호수": info.get("rooms", []),
            "요청건수": info.get("request_count", 0),
            "수량합": info.get("total_quantity", 0),
            "최대필요인원수": info.get("max_required_people", 0),
        })

    return {
        "건물명": result["building"],
        "상태": result["status"],
        "시작노드": result["start_node"],
        "투입인원수": result["dispatch_people"],
        "총방문노드순서": result["full_node_path"],
        "수거노드순서": result["pickup_node_order"],
        "수거호수순서": result["pickup_room_order"],
        "수거노드": pickup_nodes,
    }


def flatten_full_node_path(result: dict) -> list[dict]:
    rows = []
    node_info = result.get("node_info", {})

    for seq, node in enumerate(result.get("full_node_path", []), start=1):
        is_pickup = node in node_info
        rooms = node_info.get(node, {}).get("rooms", [])
        rows.append({
            "building": result["building"],
            "seq": seq,
            "node": node,
            "is_pickup_node": is_pickup,
            "pickup_rooms": ",".join(map(str, rooms)) if is_pickup else "",
        })

    return rows


def flatten_pickup_nodes(result: dict) -> list[dict]:
    rows = []
    node_info = result.get("node_info", {})

    for seq, node in enumerate(result.get("pickup_node_order", []), start=1):
        info = node_info.get(node, {})
        rows.append({
            "building": result["building"],
            "visit_seq": seq,
            "pickup_node": node,
            "pickup_rooms": ",".join(map(str, info.get("rooms", []))),
            "request_count": info.get("request_count", 0),
            "total_quantity": info.get("total_quantity", 0),
            "max_required_people": info.get("max_required_people", 0),
        })

    return rows


def flatten_segments(result: dict) -> list[dict]:
    rows = []
    for seg in result.get("detailed", []):
        for edge_no, e in enumerate(seg["path_edges"], start=1):
            rows.append({
                "building": result["building"],
                "algorithm": result["algorithm"],
                "segment_no": seg["segment_no"],
                "edge_no_in_segment": edge_no,
                "from_visit_node": seg["from_visit_node"],
                "to_visit_node": seg["to_visit_node"],
                "from_rooms": ",".join(map(str, seg["from_rooms"])),
                "to_rooms": ",".join(map(str, seg["to_rooms"])),
                "path_nodes": " -> ".join(seg["path_nodes"]),
                "from_node": e["from"],
                "to_node": e["to"],
                "edge_id": e["edge_id"],
                "edge_type": e["edge_type"],
                "length": e["length"],
                "edge_cost": e["edge_cost"],
                "segment_cost": seg["segment_cost"],
                "segment_stair_edges": seg["segment_stair_edges"],
                "segment_consecutive_elevators": seg["segment_consecutive_elevators"],
                "total_cost": result["total_cost"],
            })
    return rows


# ============================================================
# 콘솔 출력 및 저장
# ============================================================
def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"건물: {result['building']}")
    print("=" * 60)

    if result["status"] != "ok":
        print(f"상태: {result['status']}")
        if "unmatched" in result and len(result["unmatched"]) > 0:
            cols = [c for c in ["설치장소", "_room"] if c in result["unmatched"].columns]
            print("\n매핑 안 된 요청:")
            print(result["unmatched"][cols])
        return

    print(f"시작 노드: {result['start_node']}")
    print(f"투입인원수: {result['dispatch_people']}")
    print(f"알고리즘: {result['algorithm']}")
    print(f"총 비용: {result['total_cost']:.2f}")
    print(f"사용 계단 edge 수: {result['total_stair_edges']}")
    print(f"연속 elevator 발생 수: {result['total_consecutive_elevators']}")

    print("\n총 방문 노드 순서")
    print(" -> ".join(result["full_node_path"]) if result["full_node_path"] else "이동 노드 없음")

    print("\n수거 노드")
    for node in result["pickup_node_order"]:
        info = result["node_info"][node]
        print(
            f"- 수거노드={node} | 호수={info['rooms']} | "
            f"요청건수={info['request_count']} | 수량합={info['total_quantity']} | "
            f"최대필요인원={info['max_required_people']}"
        )

    if len(result["unmatched"]) > 0:
        print("\n매핑 안 된 요청")
        cols = [c for c in ["설치장소", "_room"] if c in result["unmatched"].columns]
        print(result["unmatched"][cols])


def save_outputs(results: list[dict], output_dir: str | Path, raw_input: dict):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    public_results = [result_to_public_dict(r) for r in results]
    output_json = {
        "입력": raw_input,
        "건물별경로": public_results,
    }

    with open(output_dir / "building_routes_result.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    summary_rows = []
    full_node_path_rows = []
    pickup_node_rows = []
    detail_rows = []
    unmatched_rows = []

    for r in results:
        if r["status"] == "ok":
            summary_rows.append({
                "building": r["building"],
                "status": r["status"],
                "start_node": r["start_node"],
                "dispatch_people": r["dispatch_people"],
                "algorithm": r["algorithm"],
                "full_node_path": " -> ".join(r["full_node_path"]),
                "pickup_node_order": " -> ".join(r["pickup_node_order"]),
                "pickup_room_order": " -> ".join(r["pickup_room_order"]),
                "total_cost": r["total_cost"],
                "total_stair_edges": r["total_stair_edges"],
                "total_consecutive_elevators": r["total_consecutive_elevators"],
                "stair_policy": json.dumps(r["stair_policy"], ensure_ascii=False),
            })
            full_node_path_rows.extend(flatten_full_node_path(r))
            pickup_node_rows.extend(flatten_pickup_nodes(r))
            detail_rows.extend(flatten_segments(r))
        else:
            summary_rows.append({
                "building": r["building"],
                "status": r["status"],
            })

        if "unmatched" in r and len(r["unmatched"]) > 0:
            temp = r["unmatched"].copy()
            temp["_result_building"] = r["building"]
            unmatched_rows.append(temp)

    pd.DataFrame(summary_rows).to_csv(output_dir / "building_route_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_node_path_rows).to_csv(output_dir / "building_full_node_path.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pickup_node_rows).to_csv(output_dir / "building_pickup_nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(output_dir / "building_route_details.csv", index=False, encoding="utf-8-sig")

    if unmatched_rows:
        pd.concat(unmatched_rows, ignore_index=True).to_csv(output_dir / "unmatched_requests.csv", index=False, encoding="utf-8-sig")


# ============================================================
# main (파일 직접 실행 시)
# ============================================================
def main(params: dict = PARAMS):
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
            print(f"\n{building} 처리 중 오류: {e}")

        results.append(result)
        print_result(result)

    save_outputs(results, params["output_dir"], meta["raw_input"])

    print("\n" + "=" * 60)
    print(f"결과 저장 완료: {params['output_dir']}")
    print("=" * 60)


# ============================================================
# API 진입점 (api.py → run_optimizer 호출용)
# ============================================================
def run_optimizer(df: pd.DataFrame, _df_avail: pd.DataFrame) -> list[dict]:
    """
    api.py에서 호출하는 진입점.
    df 컬럼: 신청번호, 신청일자, 신청부서, 품명, 설치장소, 수량, 필요인원수
    """
    dispatch_people = float(df["필요인원수"].max()) if "필요인원수" in df.columns else PARAMS["default_dispatch_people"]

    schedule = df.copy().reset_index(drop=True)
    schedule["_input_row_id"] = schedule.index + 1

    parsed = schedule["설치장소"].apply(split_location_to_building_room)
    schedule["_building"]         = [p[0] for p in parsed]
    schedule["_room"]             = [p[1] for p in parsed]
    schedule["_quantity"]         = schedule["수량"].apply(lambda x: to_float(x, 1))
    schedule["_required_people"]  = schedule["필요인원수"].apply(lambda x: to_float(x, 1))
    schedule["_dispatch_people"]  = dispatch_people

    output_rows = []
    for building, building_items in schedule.groupby("_building"):
        try:
            result = solve_building(building, building_items.reset_index(drop=True), dispatch_people, PARAMS)
        except Exception as e:
            print(f"[오류] {building} 처리 실패: {e}")
            continue

        if result["status"] != "ok":
            continue

        node_info = result["node_info"]
        visited = set()
        for node in result["pickup_node_order"]:
            if node in visited:
                continue
            visited.add(node)

            info = node_info.get(node, {})
            for row_id in info.get("input_row_ids", []):
                orig = schedule[schedule["_input_row_id"] == row_id]
                if orig.empty:
                    continue
                row = orig.iloc[0]
                output_rows.append({
                    "출동일시":   "-",
                    "신청서번호": str(row.get("신청번호", "-")),
                    "신청부서":   str(row.get("신청부서", "-")),
                    "설치장소":   str(row.get("설치장소", "-")),
                    "품명":       str(row.get("품명", "-")),
                    "수량":       int(to_float(row.get("수량", 1), 1)),
                    "가용명단":   "-",
                    "투입인원수": int(dispatch_people),
                })

    return output_rows


if __name__ == "__main__":
    main()

