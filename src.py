
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
    segs = []
    for region in regions[: len(seeds)]:
        poly = [vertices[v] for v in region]
        clipped = rectangle_clip_polygon(poly, 0.0, width, 0.0, height)
        if len(clipped) < 2:
            continue
        cyc = clipped[1:] + clipped[:1]
        for a, b in zip(clipped, cyc):
            if np.linalg.norm(a - b) > params.min_edge_len * 0.5:
                segs.append((np.asarray(a), np.asarray(b)))
    nodes, edges = deduplicate_segments(segs, tol=1e-5, min_len=params.min_edge_len)
    z_levels = np.linspace(0.0, height, params.n_stories + 1)
    nodes, edges = split_edges_at_story_levels(nodes, edges, z_levels, tol=1e-6)
    nodes, edges, prune_stats = prune_face_graph_to_base(nodes, edges, z_levels, tol=1e-6)
    floor_nodes = collect_floor_node_indices(nodes, z_levels, tol=1e-6)
    return {
        "seeds": seeds,
        "face_nodes": nodes,
        "face_edges": edges,
        "floor_nodes": floor_nodes,
        "width": width,
        "height": height,
        "prune_stats": prune_stats,
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


def _local_axes_from_segment(p1_2d: np.ndarray, p2_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1 = np.array([float(p1_2d[0]), float(p1_2d[1]), 0.0], dtype=float)
    p2 = np.array([float(p2_2d[0]), float(p2_2d[1]), 0.0], dtype=float)
    vx = p2 - p1
    L = np.linalg.norm(vx)
    if L < EPS:
        raise ValueError('Elemento di lunghezza quasi nulla')
    vx = vx / L
    vy = np.array([0.0, 0.0, 1.0], dtype=float)
    vz = np.cross(vx, vy)
    nz = np.linalg.norm(vz)
    if nz < EPS:
        raise ValueError('Impossibile costruire vz')
    vz = vz / nz
    return p1, p2, vx, vz


def build_opensees_face_model(geometry: Dict[str, object], load_case: str = 'combined', do_eigen: bool = True) -> Dict[str, object]:
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

    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    for tag, (x, z) in enumerate(face_nodes, start=1):
        ops.node(tag, float(x), float(z), 0.0)

    base_node_tags = []
    constrained_planar_tags = []
    for tag, (_, z) in enumerate(face_nodes, start=1):
        if abs(float(z)) < 1e-8:
            ops.fix(tag, 1, 1, 1, 1, 1, 1)
            base_node_tags.append(tag)
        else:
            # Telaio 3D planare nel piano XY: liberi UX, UY, RZ; vincolati UZ, RX, RY
            ops.fix(tag, 0, 0, 1, 1, 1, 0)
            constrained_planar_tags.append(tag)

    ele_pairs = []
    ele_lengths = []
    elem_local_axes = []
    ele_tag = 1
    transf_tag = 1
    for i, j in face_edges:
        ni = int(i) + 1
        nj = int(j) + 1
        p1_2d = face_nodes[int(i)]
        p2_2d = face_nodes[int(j)]
        try:
            p1_3d, p2_3d, vx, vecxz = _local_axes_from_segment(p1_2d, p2_2d)
        except Exception:
            continue
        L = float(np.linalg.norm(p2_3d - p1_3d))
        if L < params.min_edge_len * 0.5:
            continue
        ops.geomTransf('Linear', transf_tag, float(vecxz[0]), float(vecxz[1]), float(vecxz[2]))
        ops.element('elasticBeamColumn', ele_tag, ni, nj, exo['A'], params.steel_E, params.steel_G, exo['J'], exo['Iy'], exo['Iz'], transf_tag, '-mass', exo['A'] * params.steel_rho)
        ele_pairs.append((ni, nj))
        ele_lengths.append(L)
        elem_local_axes.append({'vx': vx.tolist(), 'vy': [0.0, 0.0, 1.0], 'vz': vecxz.tolist()})
        ele_tag += 1
        transf_tag += 1

    floor_area_face = max((params.plan_size ** 2 - params.core_size ** 2) / 4.0, 1.0)
    g_floor = params.floor_dead_kN_m2 * 1e3 * floor_area_face
    q_floor = params.floor_live_kN_m2 * 1e3 * floor_area_face
    floor_mass = (g_floor + 0.3 * q_floor) / 9.81
    wind_story_face = 0.5 * params.wind_line_kN_m * 1e3 * params.story_height

    for k in range(1, len(z_levels)):
        story_nodes = [int(i) + 1 for i in floor_nodes.get(k, [])]
        if not story_nodes:
            continue
        mx = floor_mass / len(story_nodes)
        my = mx if params.include_vertical_mass else 1e-9
        for tag in story_nodes:
            ops.mass(tag, mx, my, 1e-9, 1e-9, 1e-9, 1e-9)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for k in range(1, len(z_levels)):
        story_nodes = [int(i) + 1 for i in floor_nodes.get(k, [])]
        if not story_nodes:
            continue
        if load_case in ('lateral', 'combined'):
            px = wind_story_face / len(story_nodes)
            for tag in story_nodes:
                ops.load(tag, px, 0.0, 0.0, 0.0, 0.0, 0.0)
        if load_case in ('gravity', 'combined'):
            py = -g_floor / len(story_nodes)
            for tag in story_nodes:
                ops.load(tag, 0.0, py, 0.0, 0.0, 0.0, 0.0)

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-8, 50)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ok = ops.analyze(1)

    node_disp = np.zeros((len(face_nodes), 6), dtype=float)
    for i in range(len(face_nodes)):
        try:
            node_disp[i, :] = np.asarray(ops.nodeDisp(i + 1), dtype=float)
        except Exception:
            node_disp[i, :] = 0.0

    elem_forces = []
    elem_quantities = []
    for etag in range(1, len(ele_pairs) + 1):
        try:
            f = np.asarray(ops.eleResponse(etag, 'force'), dtype=float)
        except Exception:
            f = np.zeros(12, dtype=float)
        elem_forces.append(f)
        if len(f) >= 12:
            N = max(abs(float(f[0])), abs(float(f[6])))
            V = max(abs(float(f[1])), abs(float(f[7])))
            M = max(abs(float(f[5])), abs(float(f[11])))
        else:
            N = V = M = 0.0
        elem_quantities.append({'N': N, 'V': V, 'M': M})

    eig_vals = []
    periods = []
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
    return {
        'analysis_ok': int(ok) == 0,
        'load_case': load_case,
        'nodes': face_nodes,
        'edges': np.asarray(ele_pairs, dtype=int) - 1,
        'base_node_tags': base_node_tags,
        'constrained_planar_tags': constrained_planar_tags,
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
