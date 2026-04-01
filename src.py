
"""src.py
Pipeline focalizzata su:
1) generazione e analisi FEM della sola facciata 2D Voronoi;
2) costruzione del modello geometrico 3D a partire dalla facciata 2D.

Correzioni di stabilità applicate:
- pruning dei componenti scollegati dalla base;
- rimozione di nodi orfani (non appartenenti ad alcuna asta);
- modello OpenSees eseguito come telaio 3D planare, ma con vincoli ai gradi di libertà
  fuori piano: per tutti i nodi non di base si impone UZ=0, RX=0, RY=0; restano liberi UX, UY, RZ.

Questo evita i modi di corpo rigido fuori piano che rendono la matrice singolare.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple, Optional
import math
import json
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import Voronoi

EPS = 1e-9


@dataclass
class TowerParams:
    plan_size: float = 53.0
    total_height: float = 351.0
    n_stories: int = 90
    core_size: float = 25.9
    n_seeds: int = 260
    random_seed: int = 7
    min_edge_len: float = 0.75
    belt_levels: Tuple[int, ...] = (30, 60)
    belt_strength: float = 2.0
    corner_strength: float = 1.8
    base_strength: float = 2.5
    mode: str = "adaptive"

    steel_E: float = 210e9
    steel_G: float = 81e9
    steel_rho: float = 7850.0
    exo_b: float = 1.10
    exo_t: float = 0.09

    floor_dead_kN_m2: float = 7.0
    floor_live_kN_m2: float = 3.0
    wind_line_kN_m: float = 200.0
    include_vertical_mass: bool = True
    n_eigen: int = 6
    use_floor_diaphragm: bool = True   # equalDOF UX ai piani nel modello 2D

    # Nucleo in c.a.
    concrete_E: float = 30e9
    concrete_nu: float = 0.2
    concrete_rho: float = 2500.0
    core_t: float = 0.50

    @property
    def story_height(self) -> float:
        return self.total_height / self.n_stories


def shs_properties(b: float, t: float, G: float) -> Dict[str, float]:
    bi = max(b - 2.0 * t, EPS)
    A = b * b - bi * bi
    Iy = (b**4 - bi**4) / 12.0
    Iz = Iy
    J = 2.0 * t * (b - t) ** 3 / 3.0
    return {"A": A, "Iy": Iy, "Iz": Iz, "J": J, "G": G}


def segment_intersection_with_x(p1: np.ndarray, p2: np.ndarray, x: float) -> np.ndarray:
    if abs(p2[0] - p1[0]) < EPS:
        return np.array([x, p1[1]])
    t = (x - p1[0]) / (p2[0] - p1[0])
    return np.array([x, p1[1] + t * (p2[1] - p1[1])])


def segment_intersection_with_y(p1: np.ndarray, p2: np.ndarray, y: float) -> np.ndarray:
    if abs(p2[1] - p1[1]) < EPS:
        return np.array([p1[0], y])
    t = (y - p1[1]) / (p2[1] - p1[1])
    return np.array([p1[0] + t * (p2[0] - p1[0]), y])


def rectangle_clip_polygon(poly: List[np.ndarray], xmin: float, xmax: float, ymin: float, ymax: float) -> List[np.ndarray]:
    def clip_edge(vertices, inside, intersect):
        if not vertices:
            return []
        out = []
        prev = vertices[-1]
        prev_inside = inside(prev)
        for curr in vertices:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif prev_inside:
                out.append(intersect(prev, curr))
            prev, prev_inside = curr, curr_inside
        return out

    poly = [np.array(p, dtype=float) for p in poly]
    poly = clip_edge(poly, lambda p: p[0] >= xmin - EPS, lambda p1, p2: segment_intersection_with_x(p1, p2, xmin))
    poly = clip_edge(poly, lambda p: p[0] <= xmax + EPS, lambda p1, p2: segment_intersection_with_x(p1, p2, xmax))
    poly = clip_edge(poly, lambda p: p[1] >= ymin - EPS, lambda p1, p2: segment_intersection_with_y(p1, p2, ymin))
    poly = clip_edge(poly, lambda p: p[1] <= ymax + EPS, lambda p1, p2: segment_intersection_with_y(p1, p2, ymax))
    return poly


def voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
    if vor.points.shape[1] != 2:
        raise ValueError("Richiesti punti 2D")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2.0

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)


def _weight_field(x: np.ndarray, z: np.ndarray, width: float, height: float, params: TowerParams) -> np.ndarray:
    x_local = (x - width / 2.0) / (width / 2.0 + EPS)
    z_local = z / (height + EPS)
    w = np.ones_like(x, dtype=float)
    if params.mode in ("adaptive", "belts", "megaframe"):
        corner = np.abs(x_local) ** 1.5
        w *= 1.0 + params.corner_strength * corner
        base = (1.0 - z_local) ** 1.2
        w *= 1.0 + params.base_strength * base
        for lvl in params.belt_levels:
            zc = lvl * params.story_height
            sigma = 2.0 * params.story_height
            w *= 1.0 + params.belt_strength * np.exp(-0.5 * ((z - zc) / sigma) ** 2)
    if params.mode == "megaframe":
        diag1 = np.exp(-((z_local - (x / width)) ** 2) / 0.02)
        diag2 = np.exp(-((z_local - (1.0 - x / width)) ** 2) / 0.02)
        w *= 1.0 + 1.8 * np.maximum(diag1, diag2)
    if params.mode == "random":
        w = np.ones_like(x)
    return np.clip(w, 1e-6, None)


def unique_points_2d(pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    out = []
    used = set()
    for p in pts:
        key = (round(float(p[0]) / tol), round(float(p[1]) / tol))
        if key not in used:
            used.add(key)
            out.append(p)
    return np.asarray(out, dtype=float)


def sample_variable_density_seeds(width: float, height: float, params: TowerParams) -> np.ndarray:
    rng = np.random.default_rng(params.random_seed)
    pool_n = max(5000, params.n_seeds * 25)
    x = rng.random(pool_n) * width
    z = rng.random(pool_n) * height
    w = _weight_field(x, z, width, height, params)
    p = w / w.sum()
    idx = rng.choice(np.arange(pool_n), size=params.n_seeds, replace=False, p=p)
    seeds = np.column_stack([x[idx], z[idx]])
    anchors = []
    n_edge = max(10, params.n_seeds // 25)
    zs = np.linspace(0.0, height, n_edge)
    for zz in zs:
        anchors.append([0.0, zz])
        anchors.append([width, zz])
    for lvl in params.belt_levels:
        zz = min(height, max(0.0, lvl * params.story_height))
        xs = np.linspace(0.0, width, max(6, n_edge // 2))
        anchors.extend([[xx, zz] for xx in xs])
    seeds = np.vstack([seeds, np.asarray(anchors, dtype=float)])
    return unique_points_2d(seeds, tol=0.08)


def deduplicate_segments(segments: Iterable[Tuple[np.ndarray, np.ndarray]], tol: float = 1e-5, min_len: float = 0.0):
    node_map: Dict[Tuple[int, int], int] = {}
    nodes: List[np.ndarray] = []
    edges_set = set()
    def add_node(p: np.ndarray) -> int:
        key = (round(float(p[0]) / tol), round(float(p[1]) / tol))
        if key in node_map:
            return node_map[key]
        idx = len(nodes)
        nodes.append(np.asarray(p, dtype=float))
        node_map[key] = idx
        return idx
    for a, b in segments:
        if np.linalg.norm(a - b) < min_len - EPS:
            continue
        i = add_node(a)
        j = add_node(b)
        if i == j:
            continue
        edges_set.add(tuple(sorted((i, j))))
    return np.asarray(nodes, dtype=float), np.asarray(sorted(edges_set), dtype=int)


def split_edges_at_story_levels(nodes: np.ndarray, edges: np.ndarray, z_levels: np.ndarray, tol: float = 1e-8):
    new_nodes = nodes.tolist()
    new_edges = []
    for i, j in edges:
        p1 = nodes[i]
        p2 = nodes[j]
        if abs(p1[1] - p2[1]) < tol:
            new_edges.append((i, j))
            continue
        cuts = [p1, p2]
        zmin, zmax = sorted([p1[1], p2[1]])
        for zc in z_levels:
            if zmin + tol < zc < zmax - tol:
                t = (zc - p1[1]) / (p2[1] - p1[1])
                x = p1[0] + t * (p2[0] - p1[0])
                cuts.append(np.array([x, zc], dtype=float))
        cuts = np.asarray(cuts, dtype=float)
        cuts = cuts[np.argsort(np.linalg.norm(cuts - p1, axis=1))]
        idxs = []
        for pt in cuts:
            idx = len(new_nodes)
            new_nodes.append(pt)
            idxs.append(idx)
        for a, b in zip(idxs[:-1], idxs[1:]):
            if np.linalg.norm(np.asarray(new_nodes[a]) - np.asarray(new_nodes[b])) > tol:
                new_edges.append((a, b))
    return deduplicate_segments([(np.asarray(new_nodes[i]), np.asarray(new_nodes[j])) for i, j in new_edges], tol=1e-6, min_len=0.0)


def collect_floor_node_indices(nodes: np.ndarray, z_levels: np.ndarray, tol: float = 1e-6) -> Dict[int, List[int]]:
    out = {k: [] for k in range(len(z_levels))}
    for i, p in enumerate(nodes):
        for k, z in enumerate(z_levels):
            if abs(float(p[1]) - float(z)) <= tol:
                out[k].append(i)
    return out


def remove_dangling_nodes(nodes: np.ndarray, edges: np.ndarray, z_levels: np.ndarray, tol: float = 1e-6):
    """Rimuove iterativamente i nodi connessi a un solo elemento (grado 1).

    I nodi di base (z ≈ z_levels[0]) sono preservati anche se di grado 1,
    perché strutturalmente vincolati. Tutti gli altri nodi devono essere
    connessi ad almeno due elementi per garantire un'analisi FEM coerente.

    L'algoritmo è applicato in modo iterativo finché non esistono più nodi
    pendenti: rimuovendo un nodo pendente può rendere pendente il suo vicino.
    """
    base_z = float(z_levels[0])
    nodes = np.asarray(nodes, dtype=float)
    edges_set: set = set(tuple(sorted((int(i), int(j)))) for i, j in edges)

    total_removed_nodes = 0
    total_removed_edges = 0

    while True:
        # Grado di ogni nodo
        degree: Dict[int, int] = defaultdict(int)
        for i, j in edges_set:
            degree[i] += 1
            degree[j] += 1

        # Nodi di grado 1 non di base (liberi → instabili in FEM)
        dangling = {
            idx for idx in range(len(nodes))
            if degree.get(idx, 0) == 1
            and abs(float(nodes[idx][1]) - base_z) > tol
        }

        if not dangling:
            break

        edges_to_remove = {e for e in edges_set if e[0] in dangling or e[1] in dangling}
        edges_set -= edges_to_remove
        total_removed_edges += len(edges_to_remove)
        total_removed_nodes += len(dangling)

    # Rimappa gli indici dei nodi ancora attivi
    used_nodes: set = set()
    for i, j in edges_set:
        used_nodes.add(i)
        used_nodes.add(j)

    keep_nodes = sorted(used_nodes)
    mapping = {old: new for new, old in enumerate(keep_nodes)}
    new_nodes = np.asarray([nodes[i] for i in keep_nodes], dtype=float)
    new_edges = (
        np.asarray(sorted((mapping[i], mapping[j]) for i, j in edges_set), dtype=int)
        if edges_set else np.zeros((0, 2), dtype=int)
    )

    stats = {
        'dangling_nodes_removed': total_removed_nodes,
        'dangling_edges_removed': total_removed_edges,
    }
    return new_nodes, new_edges, stats


def prune_face_graph_to_base(nodes: np.ndarray, edges: np.ndarray, z_levels: np.ndarray, tol: float = 1e-6):
    """Mantiene solo i componenti connessi che contengono almeno un nodo di base.
    Rimuove anche nodi orfani (non incidenti ad alcuna asta).
    """
    n = len(nodes)
    if n == 0:
        return nodes, edges, {'components_total': 0, 'components_kept': 0, 'nodes_removed': 0, 'edges_removed': 0}

    adj = defaultdict(set)
    used_nodes = set()
    for i, j in edges:
        i = int(i); j = int(j)
        adj[i].add(j)
        adj[j].add(i)
        used_nodes.add(i)
        used_nodes.add(j)

    base_nodes = {i for i, p in enumerate(nodes) if abs(float(p[1]) - float(z_levels[0])) <= tol}
    visited = set()
    keep_nodes = set()
    components_total = 0
    components_kept = 0

    for start in sorted(used_nodes):
        if start in visited:
            continue
        components_total += 1
        q = deque([start])
        comp = set([start])
        visited.add(start)
        touches_base = start in base_nodes
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    comp.add(v)
                    q.append(v)
                    if v in base_nodes:
                        touches_base = True
        if touches_base:
            components_kept += 1
            keep_nodes |= comp

    # rimappa indici
    keep_nodes = sorted(keep_nodes)
    mapping = {old: new for new, old in enumerate(keep_nodes)}
    new_nodes = np.asarray([nodes[i] for i in keep_nodes], dtype=float)
    new_edges = []
    for i, j in edges:
        i = int(i); j = int(j)
        if i in mapping and j in mapping:
            new_edges.append((mapping[i], mapping[j]))
    new_edges = np.asarray(sorted(set(tuple(sorted(e)) for e in new_edges)), dtype=int) if new_edges else np.zeros((0,2), dtype=int)

    stats = {
        'components_total': int(components_total),
        'components_kept': int(components_kept),
        'nodes_removed': int(len(nodes) - len(new_nodes)),
        'edges_removed': int(len(edges) - len(new_edges)),
    }
    return new_nodes, new_edges, stats


def _connected_components_from_edges(n_nodes: int, edges: np.ndarray) -> List[set[int]]:
    """Restituisce tutti i componenti connessi del grafo nodi/aste."""
    if n_nodes <= 0:
        return []

    adj: Dict[int, set[int]] = defaultdict(set)
    touched: set[int] = set()
    for raw_i, raw_j in np.asarray(edges, dtype=int):
        i = int(raw_i)
        j = int(raw_j)
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)
        touched.add(i)
        touched.add(j)

    components: List[set[int]] = []
    visited: set[int] = set()
    for start in range(n_nodes):
        if start in visited:
            continue
        visited.add(start)
        if start not in touched:
            components.append({start})
            continue
        q = deque([start])
        comp = {start}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    comp.add(v)
                    q.append(v)
        components.append(comp)
    return components


def _is_perimeter_node(pt: np.ndarray, width: float, height: float, tol: float = 1e-6) -> bool:
    x = float(pt[0]); z = float(pt[1])
    return (
        abs(x - 0.0) <= tol or
        abs(x - width) <= tol or
        abs(z - 0.0) <= tol or
        abs(z - height) <= tol
    )


def build_perimeter_segments(width: float, height: float, z_levels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Costruisce il telaio perimetrale 2D della facciata:
    - montante sinistro x=0, spezzato a ogni livello di piano
    - montante destro x=width, spezzato a ogni livello di piano
    - trave di base z=0
    - trave di coronamento z=height
    """
    zvals = sorted({0.0, float(height), *[float(z) for z in np.asarray(z_levels, dtype=float)]})
    segments: List[Tuple[np.ndarray, np.ndarray]] = []

    for z1, z2 in zip(zvals[:-1], zvals[1:]):
        if abs(z2 - z1) <= EPS:
            continue
        segments.append((np.array([0.0, z1], dtype=float), np.array([0.0, z2], dtype=float)))
        segments.append((np.array([width, z1], dtype=float), np.array([width, z2], dtype=float)))

    segments.append((np.array([0.0, 0.0], dtype=float), np.array([width, 0.0], dtype=float)))
    segments.append((np.array([0.0, height], dtype=float), np.array([width, height], dtype=float)))
    return segments


def connect_face_components(nodes: np.ndarray, edges: np.ndarray, z_levels: np.ndarray, width: float, height: float, tol: float = 1e-6):
    """
    Rende la mesh un unico grafo connesso aggiungendo il numero minimo di
    aste-ponte (una per ogni componente secondario).

    Strategia di scelta della coppia di nodi da collegare:
    1. preferenza a nodi sullo stesso interpiano;
    2. preferenza a nodi di bordo/perimetro;
    3. fallback sulla distanza euclidea minima.
    """
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)

    comps = _connected_components_from_edges(len(nodes), edges)
    comps_nonempty = [c for c in comps if c]
    before = len(comps_nonempty)
    if before <= 1:
        return nodes, edges, {
            'components_before_connect': int(before),
            'components_after_connect': int(before),
            'bridges_added': 0,
        }

    base_z = float(z_levels[0]) if len(z_levels) else 0.0
    base_nodes = {i for i, p in enumerate(nodes) if abs(float(p[1]) - base_z) <= tol}

    def comp_score(comp: set[int]):
        return (len(comp & base_nodes), len(comp))

    anchor = set(max(comps_nonempty, key=comp_score))
    remaining = [set(c) for c in comps_nonempty if c is not anchor and c != anchor]

    edge_set = {tuple(sorted((int(i), int(j)))) for i, j in edges if int(i) != int(j)}
    bridges_added = 0

    def node_priority(i: int, j: int):
        pi = nodes[i]
        pj = nodes[j]
        same_level = int(any(abs(float(pi[1]) - float(z)) <= tol and abs(float(pj[1]) - float(z)) <= tol for z in z_levels))
        on_perimeter = int(_is_perimeter_node(pi, width, height, tol) or _is_perimeter_node(pj, width, height, tol))
        dist2 = float(np.sum((pi - pj) ** 2))
        return (-same_level, -on_perimeter, dist2, i, j)

    while remaining:
        best = None
        best_idx = None
        anchor_ids = sorted(anchor)
        for ridx, comp in enumerate(remaining):
            comp_ids = sorted(comp)
            for i in anchor_ids:
                for j in comp_ids:
                    cand = node_priority(i, j)
                    if best is None or cand < best:
                        best = cand
                        best_idx = ridx
        _, _, _, ni, nj = best
        edge_set.add(tuple(sorted((int(ni), int(nj)))))
        bridges_added += 1
        anchor |= remaining.pop(best_idx)

    new_edges = np.asarray(sorted(edge_set), dtype=int) if edge_set else np.zeros((0, 2), dtype=int)
    after = len(_connected_components_from_edges(len(nodes), new_edges))
    return nodes, new_edges, {
        'components_before_connect': int(before),
        'components_after_connect': int(after),
        'bridges_added': int(bridges_added),
    }


def build_face_voronoi(width: float, height: float, params: TowerParams) -> Dict[str, np.ndarray]:
    seeds = sample_variable_density_seeds(width, height, params)
    border_pts = []
    n_border = 18
    xs = np.linspace(0.0, width, n_border)
    zs = np.linspace(0.0, height, n_border)
    for xx in xs:
        border_pts += [[xx, -height * 0.06], [xx, height * 1.06]]
    for zz in zs:
        border_pts += [[-width * 0.06, zz], [width * 1.06, zz]]
    all_pts = np.vstack([seeds, np.asarray(border_pts, dtype=float)])

    vor = Voronoi(all_pts)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    voronoi_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for region in regions[: len(seeds)]:
        poly = [vertices[v] for v in region]
        clipped = rectangle_clip_polygon(poly, 0.0, width, 0.0, height)
        if len(clipped) < 2:
            continue
        cyc = clipped[1:] + clipped[:1]
        for a, b in zip(clipped, cyc):
            if np.linalg.norm(a - b) > params.min_edge_len * 0.5:
                voronoi_segments.append((np.asarray(a, dtype=float), np.asarray(b, dtype=float)))

    z_levels = np.linspace(0.0, height, params.n_stories + 1)
    perimeter_segments = build_perimeter_segments(width, height, z_levels)
    all_segments = voronoi_segments + perimeter_segments

    nodes, edges = deduplicate_segments(all_segments, tol=1e-5, min_len=params.min_edge_len)
    nodes, edges = split_edges_at_story_levels(nodes, edges, z_levels, tol=1e-6)

    nodes, edges, connect_stats_pre = connect_face_components(nodes, edges, z_levels, width, height, tol=1e-6)
    nodes, edges, prune_stats = prune_face_graph_to_base(nodes, edges, z_levels, tol=1e-6)
    nodes, edges, dangling_stats = remove_dangling_nodes(nodes, edges, z_levels, tol=1e-6)
    nodes, edges, connect_stats_post = connect_face_components(nodes, edges, z_levels, width, height, tol=1e-6)

    floor_nodes = collect_floor_node_indices(nodes, z_levels, tol=1e-6)
    return {
        "seeds": seeds,
        "face_nodes": nodes,
        "face_edges": edges,
        "floor_nodes": floor_nodes,
        "width": width,
        "height": height,
        "prune_stats": {
            **prune_stats,
            **dangling_stats,
            **connect_stats_pre,
            "bridges_added_after_cleanup": int(connect_stats_post.get('bridges_added', 0)),
            "components_after_cleanup": int(connect_stats_post.get('components_after_connect', 0)),
            "perimeter_segments_added": int(len(perimeter_segments)),
        },
    }


def generate_face_geometry(params: TowerParams) -> Dict[str, object]:
    face = build_face_voronoi(params.plan_size, params.total_height, params)
    return {"params": asdict(params), "face": face}


def build_tower_geometry_from_face(geometry: Dict[str, object]) -> Dict[str, object]:
    params = TowerParams(**geometry['params'])
    face_nodes = np.asarray(geometry['face']['face_nodes'], dtype=float)
    face_edges = np.asarray(geometry['face']['face_edges'], dtype=int)

    def map_face_point(face_id: int, u: float, z: float) -> np.ndarray:
        h = params.plan_size / 2.0
        if face_id == 0:
            return np.array([-h + u, -h, z])
        if face_id == 1:
            return np.array([h, -h + u, z])
        if face_id == 2:
            return np.array([h - u, h, z])
        if face_id == 3:
            return np.array([-h, h - u, z])
        raise ValueError('face_id deve stare in [0,1,2,3]')

    node_map = {}
    tower_nodes = []
    tower_edges = []
    def add_node(pt: np.ndarray) -> int:
        key = tuple(int(round(c * 1e6)) for c in pt)
        if key in node_map:
            return node_map[key]
        idx = len(tower_nodes)
        tower_nodes.append(pt)
        node_map[key] = idx
        return idx

    for face_id in range(4):
        local_to_global = {}
        for i, p in enumerate(face_nodes):
            xyz = map_face_point(face_id, float(p[0]), float(p[1]))
            local_to_global[i] = add_node(xyz)
        for i, j in face_edges:
            tower_edges.append(tuple(sorted((local_to_global[int(i)], local_to_global[int(j)]))))

    hc = params.core_size / 2.0
    z_levels = np.linspace(0.0, params.total_height, params.n_stories + 1)
    core_nodes = []
    core_shells = []
    core_map = {}
    corners = [(-hc, -hc), (hc, -hc), (hc, hc), (-hc, hc)]
    for k, z in enumerate(z_levels):
        for c_idx, (x, y) in enumerate(corners):
            core_map[(k, c_idx)] = len(core_nodes)
            core_nodes.append([x, y, float(z)])
    for k in range(params.n_stories):
        for c1, c2 in [(0,1), (1,2), (2,3), (3,0)]:
            core_shells.append([core_map[(k, c1)], core_map[(k, c2)], core_map[(k+1, c2)], core_map[(k+1, c1)]])

    return {
        'params': geometry['params'],
        'tower': {'nodes': np.asarray(tower_nodes, dtype=float), 'edges': np.asarray(sorted(set(tower_edges)), dtype=int)},
        'core': {'nodes': np.asarray(core_nodes, dtype=float), 'shells': np.asarray(core_shells, dtype=int)}
    }


def build_opensees_face_model(geometry: Dict[str, object], load_case: str = 'combined', do_eigen: bool = True) -> Dict[str, object]:
    """Modello 2D planare (ndm=2, ndf=3: UX, UY, RZ).

    Il modello lavora esclusivamente sulla connettività reale della mesh 2D:
    nessun diaframma rigido di piano (equalDOF).

    Strategia anti-singolarità:
    1. Pre-filtraggio degli elementi per lunghezza minima.
    2. Ri-pruning BFS: si tengono solo i nodi raggiungibili dalla base
       attraverso gli elementi filtrati.
    3. Solo i nodi che compaiono in almeno un elemento valido vengono
       inseriti nel modello OpenSees.
    """
    try:
        import openseespy.opensees as ops
    except Exception as exc:
        raise RuntimeError("OpenSeesPy non è disponibile nell'ambiente corrente. Installa 'openseespy'.") from exc

    params = TowerParams(**geometry['params'])
    face = geometry['face']
    face_nodes = np.asarray(face['face_nodes'], dtype=float)
    face_edges = np.asarray(face['face_edges'], dtype=int)
    z_levels = np.linspace(0.0, params.total_height, params.n_stories + 1)
    floor_nodes = face.get('floor_nodes') or collect_floor_node_indices(face_nodes, z_levels, tol=1e-6)
    exo = shs_properties(params.exo_b, params.exo_t, params.steel_G)

    # ── 1. Pre-filtraggio per lunghezza ─────────────────────────────────────
    L_min = params.min_edge_len * 0.5
    candidate: List[Tuple[int, int, float]] = []
    for raw_i, raw_j in face_edges:
        fi, fj = int(raw_i), int(raw_j)
        L = float(np.linalg.norm(face_nodes[fi] - face_nodes[fj]))
        if L >= L_min:
            candidate.append((fi, fj, L))

    # ── 2. Ri-pruning BFS dalla base attraverso gli elementi filtrati ────────
    base_face_idxs = {i for i, (_, z) in enumerate(face_nodes) if abs(float(z)) < 1e-8}
    adj_cand: Dict[int, List[int]] = defaultdict(list)
    for fi, fj, _ in candidate:
        adj_cand[fi].append(fj)
        adj_cand[fj].append(fi)

    visited: set = set(base_face_idxs)
    queue: deque = deque(base_face_idxs)
    while queue:
        u = queue.popleft()
        for v in adj_cand[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    valid: List[Tuple[int, int, float]] = [
        (fi, fj, L) for fi, fj, L in candidate if fi in visited and fj in visited
    ]
    if not valid:
        return {'analysis_ok': False, 'error': 'Nessun elemento valido dopo il filtraggio. Riduci la lunghezza minima delle aste.'}

    # ── 3. Mappa face_idx → ops_tag (1-based) ───────────────────────────────
    used_face: set = set()
    for fi, fj, _ in valid:
        used_face.add(fi)
        used_face.add(fj)
    face_to_ops: Dict[int, int] = {fi: tag for tag, fi in enumerate(sorted(used_face), start=1)}
    ops_to_face: Dict[int, int] = {v: k for k, v in face_to_ops.items()}

    floor_area_face = max((params.plan_size ** 2 - params.core_size ** 2) / 4.0, 1.0)
    g_floor = params.floor_dead_kN_m2 * 1e3 * floor_area_face
    q_floor = params.floor_live_kN_m2 * 1e3 * floor_area_face
    floor_mass = (g_floor + 0.3 * q_floor) / 9.81
    wind_story_face = 0.5 * params.wind_line_kN_m * 1e3 * params.story_height

    # ── Costruzione modello OpenSees ─────────────────────────────────────────
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    for fi, ops_tag in face_to_ops.items():
        x, z = face_nodes[fi]
        ops.node(ops_tag, float(x), float(z))

    base_node_tags: List[int] = []
    for fi, ops_tag in face_to_ops.items():
        _, z = face_nodes[fi]
        if abs(float(z)) < 1e-8:
            ops.fix(ops_tag, 1, 1, 1)
            base_node_tags.append(ops_tag)

    ele_pairs_ops: List[Tuple[int, int]] = []
    ele_lengths: List[float] = []
    elem_local_axes: List[Dict] = []
    for ele_tag, (fi, fj, L) in enumerate(valid, start=1):
        ni = face_to_ops[fi]
        nj = face_to_ops[fj]
        ops.geomTransf('Linear', ele_tag)
        ops.element(
            'elasticBeamColumn', ele_tag, ni, nj,
            exo['A'], params.steel_E, exo['Iz'], ele_tag,
            '-mass', exo['A'] * params.steel_rho,
        )
        ele_pairs_ops.append((ni, nj))
        ele_lengths.append(L)
        vx = (face_nodes[fj] - face_nodes[fi]) / L
        elem_local_axes.append({'vx': vx.tolist()})

    # ── Masse nodali ─────────────────────────────────────────────────────────
    for k in range(1, len(z_levels)):
        sn = [face_to_ops[int(i)] for i in floor_nodes.get(k, []) if int(i) in face_to_ops]
        if not sn:
            continue
        mx = floor_mass / len(sn)
        my = mx if params.include_vertical_mass else 1e-9
        for tag in sn:
            ops.mass(tag, mx, my, 1e-9)

    # ── Carichi ──────────────────────────────────────────────────────────────
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for k in range(1, len(z_levels)):
        sn = [face_to_ops[int(i)] for i in floor_nodes.get(k, []) if int(i) in face_to_ops]
        if not sn:
            continue
        if load_case in ('lateral', 'combined'):
            px = wind_story_face / len(sn)
            for tag in sn:
                ops.load(tag, float(px), 0.0, 0.0)
        if load_case in ('gravity', 'combined'):
            py = -g_floor / len(sn)
            for tag in sn:
                ops.load(tag, 0.0, float(py), 0.0)

    # ── Analisi statica ──────────────────────────────────────────────────────
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-8, 50)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ok = ops.analyze(1)

    # ── Spostamenti nodali (3 DOF) ───────────────────────────────────────────
    node_disp = np.zeros((len(face_nodes), 3), dtype=float)
    for fi, ops_tag in face_to_ops.items():
        try:
            node_disp[fi, :] = np.asarray(ops.nodeDisp(ops_tag), dtype=float)
        except Exception:
            pass

    # ── Forze di elemento (6 valori: N,V,M × 2 estremità) ───────────────────
    elem_forces: List[np.ndarray] = []
    elem_quantities: List[Dict] = []
    for etag in range(1, len(valid) + 1):
        try:
            f = np.asarray(ops.eleResponse(etag, 'force'), dtype=float)
        except Exception:
            f = np.zeros(6, dtype=float)
        elem_forces.append(f)
        if len(f) >= 6:
            N = max(abs(float(f[0])), abs(float(f[3])))
            V = max(abs(float(f[1])), abs(float(f[4])))
            M = max(abs(float(f[2])), abs(float(f[5])))
        else:
            N = V = M = 0.0
        elem_quantities.append({'N': N, 'V': V, 'M': M})

    # ── Analisi modale ───────────────────────────────────────────────────────
    eig_vals: List[float] = []
    periods: List[float] = []
    if do_eigen:
        try:
            eig_vals = list(ops.eigen(params.n_eigen))
            periods = [2.0 * math.pi / math.sqrt(lmbd) if lmbd > 0 else float('nan') for lmbd in eig_vals]
        except Exception:
            eig_vals = []
            periods = []

    ux = node_disp[:, 0]
    uz = node_disp[:, 1]
    umag = np.sqrt(ux ** 2 + uz ** 2)

    # ── Drift inter-piano ────────────────────────────────────────────────────
    story_ux: Dict[int, float] = {}
    for k in range(len(z_levels)):
        idxs = [int(i) for i in floor_nodes.get(k, []) if int(i) in face_to_ops]
        if idxs:
            story_ux[k] = float(np.mean(ux[idxs]))
    drift_ratios: List[float] = []
    for k in range(1, len(z_levels)):
        if k in story_ux and k - 1 in story_ux:
            h = float(z_levels[k] - z_levels[k - 1])
            d = abs(story_ux[k] - story_ux[k - 1])
            if h > EPS:
                drift_ratios.append(d / h)
    max_drift = float(max(drift_ratios)) if drift_ratios else 0.0

    result_edges = np.array(
        [(ops_to_face[ni], ops_to_face[nj]) for ni, nj in ele_pairs_ops], dtype=int
    )

    return {
        'analysis_ok': int(ok) == 0,
        'load_case': load_case,
        'nodes': face_nodes,
        'edges': result_edges,
        'base_node_tags': base_node_tags,
        'constrained_planar_tags': [],
        'node_disp': node_disp,
        'ux': ux,
        'uz': uz,
        'umag': umag,
        'elem_forces': elem_forces,
        'elem_quantities': elem_quantities,
        'elem_lengths': ele_lengths,
        'elem_local_axes': elem_local_axes,
        'eigenvalues': eig_vals,
        'periods_s': periods,
        'top_ux': float(np.nanmax(np.abs(ux))) if len(ux) else 0.0,
        'top_umag': float(np.nanmax(np.abs(umag))) if len(umag) else 0.0,
        'max_drift_ratio': max_drift,
        'story_ux': story_ux,
        'n_active_nodes': len(used_face),
        'n_active_elements': len(valid),
    }


def _robust_vecxz(vx: np.ndarray) -> np.ndarray:
    """Versore ortogonale a vx per geomTransf 3D Linear.

    Calcola il vettore ausiliario (vecxz) richiesto da OpenSees tramite
    proiezione di Gram-Schmidt: il risultato è garantito ortogonale alla
    direzione dell'elemento e normalizzato a 1.

    Ordine di preferenza per il vettore di riferimento:
      1. Z globale [0, 0, 1]  – per elementi prevalentemente orizzontali
      2. Y globale [0, 1, 0]  – fallback se vx è quasi verticale in Z
      3. X globale [1, 0, 0]  – fallback finale

    Il vettore scelto deve formare un angolo > ~17° con vx (|cosθ| < 0.95)
    affinché la proiezione sia numericamente stabile.
    """
    vx = np.asarray(vx, dtype=float)
    vx = vx / (np.linalg.norm(vx) + 1e-16)  # normalizza per sicurezza

    for ref in (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ):
        # Gram-Schmidt: proietta ref sul piano perpendicolare a vx
        perp = ref - float(np.dot(ref, vx)) * vx
        norm = np.linalg.norm(perp)
        if norm > 0.3:           # equivalente a |cosθ| < ~0.954 con vx
            return perp / norm

    # Fallback teoricamente irraggiungibile
    return np.array([0.0, 0.0, 1.0])


def build_opensees_tower_model(
    tower_geometry: Dict[str, object],
    geometry: Dict[str, object],
    load_case: str = 'lateral',
    do_eigen: bool = True,
) -> Dict[str, object]:
    """Analisi FEM 3D della torre (ndm=3, ndf=6).

    Elementi strutturali
    --------------------
    - Esoscheletro: elasticBeamColumn 3D sulle 4 facce (acciaio SHS).
    - Nucleo: ShellMITC4 sulle 4 pareti (c.a. ElasticMembranePlateSection).

    Vincoli di piano
    ----------------
    - rigidDiaphragm(perpDirn=3, master, *slaves) a ogni livello z_k > 0:
      collega tutti i nodi esoscheletro e nucleo dello stesso piano.
      Il master è il primo nodo utile del nucleo; i rimanenti sono slaves.
      Il diaframma vincola UX, UY, RZ (moto planare rigido);
      UZ, RX, RY restano indipendenti.

    Robustezza
    ----------
    - Pre-filtraggio degli elementi esoscheletro per lunghezza minima.
    - BFS dalla base: solo nodi raggiunti da elementi validi entrano nel modello.
    """
    try:
        import openseespy.opensees as ops
    except Exception as exc:
        raise RuntimeError("OpenSeesPy non disponibile. Installa 'openseespy'.") from exc

    params = TowerParams(**geometry['params'])
    t_nodes = np.asarray(tower_geometry['tower']['nodes'], dtype=float)   # (N,3) exo
    t_edges = np.asarray(tower_geometry['tower']['edges'], dtype=int)      # (E,2)
    c_nodes = np.asarray(tower_geometry['core']['nodes'],  dtype=float)    # (M,3)
    c_shells = np.asarray(tower_geometry['core']['shells'], dtype=int)     # (S,4)

    z_levels = np.linspace(0.0, params.total_height, params.n_stories + 1)
    tol_z = 1e-4

    # ── Pre-filtraggio esoscheletro (BFS dalla base) ─────────────────────────
    L_min = params.min_edge_len * 0.5
    cand: List[Tuple[int, int, float]] = []
    for ri, rj in t_edges:
        fi, fj = int(ri), int(rj)
        L = float(np.linalg.norm(t_nodes[fi] - t_nodes[fj]))
        if L >= L_min:
            cand.append((fi, fj, L))

    base_t = {i for i, (x, y, z) in enumerate(t_nodes) if abs(float(z)) < tol_z}
    adj_t: Dict[int, List[int]] = defaultdict(list)
    for fi, fj, _ in cand:
        adj_t[fi].append(fj); adj_t[fj].append(fi)

    vis: set = set(base_t)
    q: deque = deque(base_t)
    while q:
        u = q.popleft()
        for v in adj_t[u]:
            if v not in vis:
                vis.add(v); q.append(v)

    valid: List[Tuple[int, int, float]] = [(fi, fj, L) for fi, fj, L in cand if fi in vis and fj in vis]
    if not valid:
        return {'analysis_ok': False, 'error': 'Nessun elemento esoscheletro valido nel modello 3D.'}

    used_t: set = set()
    for fi, fj, _ in valid:
        used_t.add(fi); used_t.add(fj)

    # ── Mapping indici → tag OpenSees ─────────────────────────────────────────
    t_to_ops: Dict[int, int] = {fi: tag for tag, fi in enumerate(sorted(used_t), start=1)}
    n_used_t = len(t_to_ops)
    c_offset = n_used_t  # core tags start at n_used_t+1

    def cr(i: int) -> int:
        return int(i) + c_offset + 1

    # ── Costruzione modello OpenSees ─────────────────────────────────────────
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    for fi, ops_tag in t_to_ops.items():
        x, y, z = t_nodes[fi]
        ops.node(ops_tag, float(x), float(y), float(z))
    for i, (x, y, z) in enumerate(c_nodes):
        ops.node(cr(i), float(x), float(y), float(z))

    # ── Vincoli di base ───────────────────────────────────────────────────────
    for fi, ops_tag in t_to_ops.items():
        if abs(float(t_nodes[fi][2])) < tol_z:
            ops.fix(ops_tag, 1, 1, 1, 1, 1, 1)
    for i, (x, y, z) in enumerate(c_nodes):
        if abs(float(z)) < tol_z:
            ops.fix(cr(i), 1, 1, 1, 1, 1, 1)

    # ── Esoscheletro: elasticBeamColumn 3D ───────────────────────────────────
    exo = shs_properties(params.exo_b, params.exo_t, params.steel_G)
    ele_tag = 1
    exo_ele_info: List[Tuple[int, int]] = []  # (t_idx_i, t_idx_j)

    for fi, fj, L in valid:
        vx = (t_nodes[fj] - t_nodes[fi]) / L
        vcxz = _robust_vecxz(vx)
        ops.geomTransf('Linear', ele_tag, *[float(v) for v in vcxz])
        ops.element(
            'elasticBeamColumn', ele_tag,
            t_to_ops[fi], t_to_ops[fj],
            exo['A'], params.steel_E, params.steel_G,
            exo['J'], exo['Iy'], exo['Iz'], ele_tag,
            '-mass', exo['A'] * params.steel_rho,
        )
        exo_ele_info.append((fi, fj))
        ele_tag += 1

    n_exo_ele = len(exo_ele_info)

    # ── Nucleo: ShellMITC4 ────────────────────────────────────────────────────
    ops.section(
        'ElasticMembranePlateSection', 1,
        params.concrete_E, params.concrete_nu,
        params.core_t, params.concrete_rho,
    )
    shell_start = ele_tag
    for n1, n2, n3, n4 in c_shells:
        ops.element('ShellMITC4', ele_tag, cr(int(n1)), cr(int(n2)), cr(int(n3)), cr(int(n4)), 1)
        ele_tag += 1
    n_shell_ele = ele_tag - shell_start

    # ── Masse di piano ────────────────────────────────────────────────────────
    floor_area_3d = max(params.plan_size ** 2 - params.core_size ** 2, 1.0)
    g_floor = params.floor_dead_kN_m2 * 1e3 * floor_area_3d
    q_floor = params.floor_live_kN_m2 * 1e3 * floor_area_3d
    floor_mass_tot = (g_floor + 0.3 * q_floor) / 9.81
    wind_floor_N = params.wind_line_kN_m * 1e3 * params.story_height

    floor_t_tags: Dict[int, List[int]] = {}  # k → [ops_tag, ...]
    floor_c_tags: Dict[int, List[int]] = {}
    for k, z in enumerate(z_levels):
        if abs(z) < tol_z:
            continue
        ft = [t_to_ops[fi] for fi in sorted(used_t) if abs(float(t_nodes[fi][2]) - z) <= tol_z]
        fc = [cr(i) for i, (x, y, zn) in enumerate(c_nodes) if abs(float(zn) - z) <= tol_z]
        if ft or fc:
            floor_t_tags[k] = ft
            floor_c_tags[k] = fc

    for k in floor_t_tags:
        all_fn = floor_t_tags[k] + floor_c_tags.get(k, [])
        if not all_fn:
            continue
        m_node = floor_mass_tot / len(all_fn)
        mz = m_node if params.include_vertical_mass else 1e-9
        for tag in all_fn:
            ops.mass(tag, m_node, m_node, mz, 1e-9, 1e-9, 1e-9)

    # ── Diaframmi rigidi ──────────────────────────────────────────────────────
    # perpDirn=3 → diaframma ⊥ a Z (piano orizzontale)
    # Vincola UX, UY, RZ degli slave al master; UZ, RX, RY restano liberi.
    diaphragm_master: Dict[int, int] = {}
    for k in floor_t_tags:
        all_fn = floor_t_tags[k] + floor_c_tags.get(k, [])
        if len(all_fn) < 2:
            continue
        # master: preferisci il primo nodo nucleo (più centrale); fallback su exo
        fc = floor_c_tags.get(k, [])
        master = fc[0] if fc else all_fn[0]
        slaves = [n for n in all_fn if n != master]
        ops.rigidDiaphragm(3, master, *slaves)
        diaphragm_master[k] = master

    # ── Carichi ──────────────────────────────────────────────────────────────
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for k in floor_t_tags:
        all_fn = floor_t_tags[k] + floor_c_tags.get(k, [])
        if not all_fn:
            continue
        master = diaphragm_master.get(k)
        if load_case in ('lateral', 'combined') and master:
            ops.load(master, float(wind_floor_N), 0.0, 0.0, 0.0, 0.0, 0.0)
        if load_case in ('gravity', 'combined'):
            fz = -g_floor / len(all_fn)
            for tag in all_fn:
                ops.load(tag, 0.0, 0.0, float(fz), 0.0, 0.0, 0.0)

    # ── Analisi statica ───────────────────────────────────────────────────────
    ops.constraints('Transformation')
    ops.numberer('RCM')
    try:
        ops.system('UmfPack')
    except Exception:
        ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 100)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ok = ops.analyze(1)

    # ── Spostamenti ───────────────────────────────────────────────────────────
    t_disp = np.zeros((len(t_nodes), 6), dtype=float)
    for fi, ops_tag in t_to_ops.items():
        try:
            t_disp[fi, :] = np.asarray(ops.nodeDisp(ops_tag), dtype=float)
        except Exception:
            pass

    c_disp = np.zeros((len(c_nodes), 6), dtype=float)
    for i in range(len(c_nodes)):
        try:
            c_disp[i, :] = np.asarray(ops.nodeDisp(cr(i)), dtype=float)
        except Exception:
            pass

    # ── Forze di elemento (esoscheletro) ─────────────────────────────────────
    exo_quantities: List[Dict] = []
    for etag in range(1, n_exo_ele + 1):
        try:
            f = np.asarray(ops.eleResponse(etag, 'force'), dtype=float)
        except Exception:
            f = np.zeros(12, dtype=float)
        if len(f) >= 12:
            N = max(abs(float(f[0])), abs(float(f[6])))
            V = max(abs(float(f[1])), abs(float(f[7])))
            M = max(abs(float(f[5])), abs(float(f[11])))
        else:
            N = V = M = 0.0
        exo_quantities.append({'N': N, 'V': V, 'M': M})

    # ── Analisi modale ────────────────────────────────────────────────────────
    eig_vals: List[float] = []
    periods: List[float] = []
    if do_eigen:
        try:
            eig_vals = list(ops.eigen(params.n_eigen))
            periods = [2.0 * math.pi / math.sqrt(lmbd) if lmbd > 0 else float('nan') for lmbd in eig_vals]
        except Exception:
            pass

    # ── Deriva inter-piano (direzione X) ─────────────────────────────────────
    ops_to_t = {v: k for k, v in t_to_ops.items()}
    story_ux: Dict[int, float] = {0: 0.0}
    for k in floor_t_tags:
        idxs = [ops_to_t[tag] for tag in floor_t_tags[k]]
        if idxs:
            story_ux[k] = float(np.mean(t_disp[idxs, 0]))
    drift_ratios: List[float] = []
    for k in range(1, len(z_levels)):
        if k in story_ux and k - 1 in story_ux:
            h = float(z_levels[k] - z_levels[k - 1])
            d = abs(story_ux[k] - story_ux[k - 1])
            if h > EPS:
                drift_ratios.append(d / h)
    max_drift = float(max(drift_ratios)) if drift_ratios else 0.0

    top_z = float(np.max(t_nodes[:, 2]))
    top_mask = np.abs(t_nodes[:, 2] - top_z) < tol_z
    top_ux = float(np.mean(np.abs(t_disp[top_mask, 0]))) if np.any(top_mask) else 0.0
    top_umag = float(np.mean(np.sqrt(t_disp[top_mask, 0] ** 2 + t_disp[top_mask, 1] ** 2))) if np.any(top_mask) else 0.0

    return {
        'analysis_ok': int(ok) == 0,
        'load_case': load_case,
        'tower_nodes': t_nodes,
        'valid_edges': valid,
        'core_nodes': c_nodes,
        'core_shells': c_shells,
        'tower_disp': t_disp,
        'core_disp': c_disp,
        'exo_quantities': exo_quantities,
        'eigenvalues': eig_vals,
        'periods_s': periods,
        'top_ux': top_ux,
        'top_umag': top_umag,
        'max_drift_ratio': max_drift,
        'story_ux': story_ux,
        'n_exo_nodes': len(used_t),
        'n_exo_elements': n_exo_ele,
        'n_shell_elements': n_shell_ele,
        'n_core_nodes': len(c_nodes),
    }


def _safe_min_max(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 1.0)
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def _sample_colors(colorscale: str, values: np.ndarray):
    import plotly.colors as pc
    vals = np.asarray(values, dtype=float)
    vmin, vmax = _safe_min_max(vals)
    out = []
    for v in vals:
        t = (float(v) - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        out.append(pc.sample_colorscale(colorscale, [t])[0])
    return out, vmin, vmax


def plotly_face_traces(face: Dict[str, np.ndarray]):
    import plotly.graph_objects as go
    nodes = face['face_nodes']
    edges = face['face_edges']
    seeds = face['seeds']
    x_lines, z_lines = [], []
    for i, j in edges:
        p1 = nodes[int(i)]
        p2 = nodes[int(j)]
        x_lines += [p1[0], p2[0], None]
        z_lines += [p1[1], p2[1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_lines, y=z_lines, mode='lines', name='Maglia Voronoi', line=dict(color='#1f77b4', width=1.3)))
    fig.add_trace(go.Scatter(x=seeds[:, 0], y=seeds[:, 1], mode='markers', name='Seed', marker=dict(size=3, color='#d62728')))
    fig.update_layout(title='Facciata Voronoi 2D', xaxis_title='Sviluppo facciata [m]', yaxis_title='Quota z [m]', yaxis_scaleanchor='x', template='plotly_white', height=650)
    return fig


def plotly_tower_traces(tower_geometry: Dict[str, object]):
    import plotly.graph_objects as go
    tower_nodes = tower_geometry['tower']['nodes']
    tower_edges = tower_geometry['tower']['edges']
    core_nodes = tower_geometry['core']['nodes']
    core_shells = tower_geometry['core']['shells']
    x_lines, y_lines, z_lines = [], [], []
    for i, j in tower_edges:
        p1 = tower_nodes[int(i)]
        p2 = tower_nodes[int(j)]
        x_lines += [p1[0], p2[0], None]
        y_lines += [p1[1], p2[1], None]
        z_lines += [p1[2], p2[2], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', name='Esoscheletro 3D', line=dict(color='#0a84ff', width=3)))
    cx, cy, cz = [], [], []
    for n1, n2, n3, n4 in core_shells:
        pts = [core_nodes[int(n)] for n in [n1, n2, n3, n4, n1]]
        for a, b in zip(pts[:-1], pts[1:]):
            cx += [a[0], b[0], None]
            cy += [a[1], b[1], None]
            cz += [a[2], b[2], None]
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='lines', name='Nucleo 3D', line=dict(color='#ff7f0e', width=4)))
    fig.update_layout(title='Costruzione 3D dalla facciata 2D', scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data'), template='plotly_white', height=820)
    return fig


def plotly_face_deformed_shape(face_result: Dict[str, object], scale: float = 25.0):
    import plotly.graph_objects as go
    nodes = np.asarray(face_result['nodes'], dtype=float)
    edges = np.asarray(face_result['edges'], dtype=int)
    node_disp = np.asarray(face_result['node_disp'], dtype=float)
    deformed = nodes + np.column_stack([node_disp[:, 0] * scale, node_disp[:, 1] * scale])
    umag = np.asarray(face_result['umag'], dtype=float)
    vmin, vmax = _safe_min_max(umag)
    fig = go.Figure()
    xu, zu = [], []
    for i, j in edges:
        p1 = nodes[int(i)]
        p2 = nodes[int(j)]
        xu += [p1[0], p2[0], None]
        zu += [p1[1], p2[1], None]
    fig.add_trace(go.Scatter(x=xu, y=zu, mode='lines', name='Indeformata', line=dict(color='rgba(120,120,120,0.45)', width=1)))
    xd, zd = [], []
    for i, j in edges:
        p1 = deformed[int(i)]
        p2 = deformed[int(j)]
        xd += [p1[0], p2[0], None]
        zd += [p1[1], p2[1], None]
    fig.add_trace(go.Scatter(x=xd, y=zd, mode='lines', name=f'Deformata x{scale:g}', line=dict(color='crimson', width=1.8)))
    fig.add_trace(go.Scatter(x=deformed[:, 0], y=deformed[:, 1], mode='markers', name='|u| nodi deformati', marker=dict(size=5, color=umag, colorscale='Turbo', cmin=vmin, cmax=vmax, colorbar=dict(title='|u| [m]')), customdata=np.column_stack([node_disp[:, 0], node_disp[:, 1], umag]), hovertemplate='x=%{x:.2f} m<br>z=%{y:.2f} m<br>ux=%{customdata[0]:.4e} m<br>uz=%{customdata[1]:.4e} m<br>|u|=%{customdata[2]:.4e} m<extra></extra>'))
    fig.update_layout(title='Facciata 2D – mesh indeformata/deformata', xaxis_title='Sviluppo facciata [m]', yaxis_title='Quota z [m]', yaxis_scaleanchor='x', template='plotly_white', height=760, legend=dict(orientation='h', y=1.02))
    return fig


def plotly_face_force_map(face_result: Dict[str, object], quantity: str = 'M', deformed: bool = False, scale: float = 25.0):
    import plotly.graph_objects as go
    nodes = np.asarray(face_result['nodes'], dtype=float)
    edges = np.asarray(face_result['edges'], dtype=int)
    node_disp = np.asarray(face_result['node_disp'], dtype=float)
    mesh = nodes + np.column_stack([node_disp[:, 0] * scale, node_disp[:, 1] * scale]) if deformed else nodes
    vals = np.asarray([float(q.get(quantity, 0.0)) for q in face_result['elem_quantities']], dtype=float)
    colors, vmin, vmax = _sample_colors('Turbo', vals)
    fig = go.Figure()
    for eidx, (i, j) in enumerate(edges):
        p1 = mesh[int(i)]
        p2 = mesh[int(j)]
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines', line=dict(color=colors[eidx], width=2.5), showlegend=False, customdata=[[vals[eidx], int(i), int(j)], [vals[eidx], int(i), int(j)]], hovertemplate=(f'{quantity}=' + '%{customdata[0]:.4e}<br>' + 'i=%{customdata[1]} j=%{customdata[2]}<extra></extra>')))
    mid = mesh.mean(axis=0)
    fig.add_trace(go.Scatter(x=[mid[0]], y=[mid[1]], mode='markers', showlegend=False, marker=dict(size=0.1, color=[vmin, vmax], colorscale='Turbo', cmin=vmin, cmax=vmax, colorbar=dict(title=quantity)), hoverinfo='skip'))
    suffix = 'deformata' if deformed else 'indeformata'
    fig.update_layout(title=f'Facciata 2D – mappa {quantity} sulla mesh {suffix}', xaxis_title='Sviluppo facciata [m]', yaxis_title='Quota z [m]', yaxis_scaleanchor='x', template='plotly_white', height=760)
    return fig


def plotly_face_displacement_map(face_result: Dict[str, object], component: str = 'ux'):
    import plotly.graph_objects as go
    nodes = np.asarray(face_result['nodes'], dtype=float)
    edges = np.asarray(face_result['edges'], dtype=int)
    if component not in ('ux', 'uz', 'umag'):
        component = 'ux'
    vals = np.asarray(face_result[component], dtype=float)
    vmin, vmax = _safe_min_max(vals)
    x_lines, z_lines = [], []
    for i, j in edges:
        p1 = nodes[int(i)]
        p2 = nodes[int(j)]
        x_lines += [p1[0], p2[0], None]
        z_lines += [p1[1], p2[1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_lines, y=z_lines, mode='lines', name='Mesh', line=dict(color='rgba(90,90,90,0.30)', width=1.2)))
    fig.add_trace(go.Scatter(x=nodes[:, 0], y=nodes[:, 1], mode='markers', name=component, marker=dict(size=6, color=vals, colorscale='Turbo', cmin=vmin, cmax=vmax, colorbar=dict(title=component)), customdata=np.column_stack([face_result['ux'], face_result['uz'], face_result['umag']]), hovertemplate='x=%{x:.2f} m<br>z=%{y:.2f} m<br>ux=%{customdata[0]:.4e} m<br>uz=%{customdata[1]:.4e} m<br>|u|=%{customdata[2]:.4e} m<extra></extra>'))
    fig.update_layout(title=f'Facciata 2D – mappa spostamenti nodali ({component})', xaxis_title='Sviluppo facciata [m]', yaxis_title='Quota z [m]', yaxis_scaleanchor='x', template='plotly_white', height=760)
    return fig


def plotly_tower_deformed_traces(tower_result: Dict[str, object], scale: float = 10.0):
    """Deformata 3D dell'esoscheletro colorata per |u| nodale."""
    import plotly.graph_objects as go

    t_nodes = np.asarray(tower_result['tower_nodes'], dtype=float)
    valid_edges = tower_result['valid_edges']        # list of (fi, fj, L)
    t_disp = np.asarray(tower_result['tower_disp'], dtype=float)
    c_nodes = np.asarray(tower_result['core_nodes'], dtype=float)
    c_shells = np.asarray(tower_result['core_shells'], dtype=int)
    c_disp = np.asarray(tower_result['core_disp'], dtype=float)

    deformed_t = t_nodes.copy()
    deformed_t[:, 0] += t_disp[:, 0] * scale
    deformed_t[:, 1] += t_disp[:, 1] * scale
    deformed_t[:, 2] += t_disp[:, 2] * scale

    # Linee indeformate (grigio trasparente)
    xu, yu, zu = [], [], []
    for fi, fj, _ in valid_edges:
        p1, p2 = t_nodes[fi], t_nodes[fj]
        xu += [p1[0], p2[0], None]; yu += [p1[1], p2[1], None]; zu += [p1[2], p2[2], None]

    # Linee deformate (rosso)
    xd, yd, zd = [], [], []
    for fi, fj, _ in valid_edges:
        p1, p2 = deformed_t[fi], deformed_t[fj]
        xd += [p1[0], p2[0], None]; yd += [p1[1], p2[1], None]; zd += [p1[2], p2[2], None]

    # Nucleo: pareti deformate
    deformed_c = c_nodes.copy()
    deformed_c[:, 0] += c_disp[:, 0] * scale
    deformed_c[:, 1] += c_disp[:, 1] * scale
    deformed_c[:, 2] += c_disp[:, 2] * scale
    xc, yc, zc = [], [], []
    for n1, n2, n3, n4 in c_shells:
        pts = [deformed_c[int(n)] for n in [n1, n2, n3, n4, n1]]
        for a, b in zip(pts[:-1], pts[1:]):
            xc += [a[0], b[0], None]; yc += [a[1], b[1], None]; zc += [a[2], b[2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xu, y=yu, z=zu, mode='lines',
                               name='Indeformata', line=dict(color='rgba(180,180,180,0.35)', width=1)))
    fig.add_trace(go.Scatter3d(x=xd, y=yd, z=zd, mode='lines',
                               name=f'Esoscheletro ×{scale:g}', line=dict(color='#e63946', width=2)))
    fig.add_trace(go.Scatter3d(x=xc, y=yc, z=zc, mode='lines',
                               name=f'Nucleo ×{scale:g}', line=dict(color='#f4a261', width=3)))
    fig.update_layout(
        title=f'Torre 3D – deformata ×{scale:g} (caso: {tower_result.get("load_case","?")})',
        scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data'),
        template='plotly_white', height=850,
    )
    return fig


def plotly_tower_force_map(tower_result: Dict[str, object], quantity: str = 'N'):
    """Mappa delle forze dell'esoscheletro 3D."""
    import plotly.graph_objects as go

    t_nodes = np.asarray(tower_result['tower_nodes'], dtype=float)
    valid_edges = tower_result['valid_edges']
    exo_q = tower_result['exo_quantities']
    vals = np.asarray([float(q.get(quantity, 0.0)) for q in exo_q], dtype=float)
    colors, vmin, vmax = _sample_colors('Turbo', vals)

    fig = go.Figure()
    for eidx, (fi, fj, _) in enumerate(valid_edges):
        p1, p2 = t_nodes[fi], t_nodes[fj]
        v = vals[eidx]
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', showlegend=False,
            line=dict(color=colors[eidx], width=2),
            hovertemplate=f'{quantity}={v:.4e}<extra></extra>',
        ))
    # Colorbar ghost trace
    fig.add_trace(go.Scatter3d(
        x=[t_nodes[0, 0]], y=[t_nodes[0, 1]], z=[t_nodes[0, 2]],
        mode='markers', showlegend=False,
        marker=dict(size=0.1, color=[vmin, vmax], colorscale='Turbo',
                    cmin=vmin, cmax=vmax, colorbar=dict(title=quantity)),
        hoverinfo='skip',
    ))
    fig.update_layout(
        title=f'Esoscheletro 3D – mappa {quantity}',
        scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data'),
        template='plotly_white', height=800,
    )
    return fig


def plotly_tower_drift_profile(tower_result: Dict[str, object], story_height: float):
    """Profilo di spostamento e drift inter-piano del modello 3D."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    story_ux = tower_result.get('story_ux', {})
    if not story_ux:
        return go.Figure()
    levels = sorted(story_ux.keys())
    z_vals = [k * story_height for k in levels]
    ux_vals = [story_ux[k] for k in levels]
    drift_vals, drift_z = [], []
    for k in range(1, len(levels)):
        h = story_height
        d = abs(ux_vals[k] - ux_vals[k - 1])
        drift_vals.append(d / h if h > EPS else 0.0)
        drift_z.append((z_vals[k] + z_vals[k - 1]) / 2.0)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=['Spost. medio di piano [m]', 'Drift inter-piano [−]'])
    fig.add_trace(go.Scatter(x=ux_vals, y=z_vals, mode='lines+markers', name='ux medio',
                             line=dict(color='#0a84ff', width=2), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=drift_vals, y=drift_z, mode='lines+markers', name='drift 3D',
                             line=dict(color='#ff3b30', width=2), marker=dict(size=4)), row=1, col=2)
    fig.add_vline(x=story_height / 500.0, line_dash='dot', line_color='orange',
                  annotation_text='h/500', row=1, col=2)
    fig.update_layout(title='Profilo deriva 3D', template='plotly_white', height=600)
    fig.update_yaxes(title_text='Quota z [m]', row=1, col=1)
    return fig


def plotly_face_drift_profile(face_result: Dict[str, object], n_stories: int, story_height: float):
    """Profilo degli spostamenti medi di piano e del drift inter-piano."""
    import plotly.graph_objects as go
    story_ux = face_result.get('story_ux', {})
    if not story_ux:
        return go.Figure()
    levels = sorted(story_ux.keys())
    z_vals = [k * story_height for k in levels]
    ux_vals = [story_ux[k] for k in levels]
    drift_vals = []
    drift_z = []
    for k in range(1, len(levels)):
        h = story_height
        d = abs(ux_vals[k] - ux_vals[k - 1])
        drift_vals.append(d / h if h > EPS else 0.0)
        drift_z.append((z_vals[k] + z_vals[k - 1]) / 2.0)

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=['Spostamento medio di piano [m]', 'Drift inter-piano [−]'])
    fig.add_trace(go.Scatter(x=ux_vals, y=z_vals, mode='lines+markers', name='ux medio',
                             line=dict(color='#0a84ff', width=2), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=drift_vals, y=drift_z, mode='lines+markers', name='drift',
                             line=dict(color='#ff3b30', width=2), marker=dict(size=4)), row=1, col=2)
    # Limite H/500
    limit = story_height / 500.0
    fig.add_vline(x=limit, line_dash='dot', line_color='orange', annotation_text='H/500', row=1, col=2)
    fig.update_layout(title='Profilo spostamenti e drift inter-piano', template='plotly_white', height=600)
    fig.update_yaxes(title_text='Quota z [m]', row=1, col=1)
    return fig


def export_geometry_json(geometry: Dict[str, object]) -> str:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    return json.dumps(convert(geometry), indent=2)
