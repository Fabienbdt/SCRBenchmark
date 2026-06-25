"""
ASCDT clustering (Adaptive Spatial Clustering based on Delaunay Triangulation).

Implementation of:
  Deng M., Liu Q., Cheng T., Shi Y. (2011).
  "An adaptive spatial clustering algorithm based on Delaunay triangulation."
  Computers, Environment and Urban Systems, 35, 320-332.

Aligned with both the paper (Eq. 1-13, Def. 1-13) and the authors' MATLAB
reference code (ASCDT.m).

Algorithm (Fig. 6):
  Step 1 – Global cut:  remove Global_Long_Edges  (§3.2, Eq. 6-8)
  Step 2 – Local cut:   remove Local_Long_Edges   (§3.3, Eq. 4-5, 9-10)
  Step 3 – Chain/neck:  remove Local_Link_Edges   (§3.3, Observation 1 & Def. 10-13)
                         (optional – not in MATLAB reference code)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EdgeGraph:
    """Undirected edge graph with Euclidean lengths."""

    edges: np.ndarray    # (n_edges, 2) int64
    lengths: np.ndarray  # (n_edges,) float64

    def empty(self) -> bool:
        return self.edges.size == 0



# ---------------------------------------------------------------------------
# Delaunay triangulation
# ---------------------------------------------------------------------------

def _build_delaunay(points: np.ndarray, random_state: int = 42) -> EdgeGraph:
    """Build unique undirected Delaunay edges with Euclidean lengths."""
    from scipy.spatial import Delaunay

    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 2:
        return EdgeGraph(np.empty((0, 2), dtype=np.int64), np.empty(0))
    if n == 2:
        return EdgeGraph(
            np.array([[0, 1]], dtype=np.int64),
            np.linalg.norm(pts[0] - pts[1]).reshape(1),
        )

    tri = None
    try:
        tri = Delaunay(pts, qhull_options="QJ")
    except Exception:
        pass
    if tri is None:
        rng = np.random.default_rng(random_state)
        tri = Delaunay(pts + 1e-10 * rng.standard_normal(pts.shape), qhull_options="QJ")

    s = np.asarray(tri.simplices, dtype=np.int64)
    if s.size == 0:
        return EdgeGraph(np.empty((0, 2), dtype=np.int64), np.empty(0))

    edges = np.unique(
        np.vstack([
            np.sort(s[:, [0, 1]], 1),
            np.sort(s[:, [0, 2]], 1),
            np.sort(s[:, [1, 2]], 1),
        ]),
        axis=0,
    ).astype(np.int64)
    d = pts[edges[:, 0]] - pts[edges[:, 1]]
    return EdgeGraph(edges, np.sqrt((d * d).sum(1)))


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _vertices(graph: EdgeGraph) -> np.ndarray:
    if graph.empty():
        return np.empty(0, dtype=np.int64)
    return np.unique(graph.edges.ravel())


def _adjacency(verts: np.ndarray, edges: np.ndarray) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {int(v): set() for v in verts}
    for u, v in edges:
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))
    return adj


def _incidence(verts: np.ndarray, edges: np.ndarray) -> Dict[int, List[int]]:
    """Map vertex -> list of incident edge indices."""
    inc: Dict[int, List[int]] = {int(v): [] for v in verts}
    for i, (u, v) in enumerate(edges):
        inc.setdefault(int(u), []).append(i)
        inc.setdefault(int(v), []).append(i)
    return inc


def _paper_std(arr: np.ndarray) -> float:
    """Standard deviation with (N-1) denominator (paper Eq. 3-4)."""
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _connected_components(verts: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
    if verts.size == 0:
        return []
    adj = _adjacency(verts, edges)
    seen: Set[int] = set()
    comps: List[np.ndarray] = []
    for v in verts:
        vi = int(v)
        if vi in seen:
            continue
        stack, comp = [vi], []
        seen.add(vi)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(np.array(comp, dtype=np.int64))
    return comps


def _split(graph: EdgeGraph) -> List[EdgeGraph]:
    """Split into connected-component subgraphs (both endpoints in component)."""
    if graph.empty():
        return []
    comps = _connected_components(_vertices(graph), graph.edges)
    subs: List[EdgeGraph] = []
    for comp in comps:
        s = set(int(v) for v in comp)
        mask = np.array(
            [(int(u) in s and int(v) in s) for u, v in graph.edges],
            dtype=bool,
        )
        if mask.any():
            subs.append(EdgeGraph(graph.edges[mask], graph.lengths[mask]))
    return subs


def _prune_small_components(graph: EdgeGraph, minpts: int) -> EdgeGraph:
    """Remove connected components with <= minpts vertices (MATLAB MINPTS=5).

    Matches the MATLAB reference code which prunes small clusters after each
    decomposition step, preventing small fragments from contaminating statistics.
    """
    if graph.empty() or minpts < 1:
        return graph

    comps = _connected_components(_vertices(graph), graph.edges)
    # Collect vertices belonging to small components.
    remove_verts: Set[int] = set()
    for comp in comps:
        if len(comp) <= minpts:
            remove_verts.update(int(v) for v in comp)

    if not remove_verts:
        return graph

    # Remove all edges touching any vertex in a small component.
    keep = np.array(
        [int(u) not in remove_verts and int(v) not in remove_verts
         for u, v in graph.edges],
        dtype=bool,
    )
    if not keep.any():
        return EdgeGraph(np.empty((0, 2), dtype=np.int64), np.empty(0))
    return EdgeGraph(graph.edges[keep], graph.lengths[keep])


# ---------------------------------------------------------------------------
# K-order neighbourhood helpers (Def. 1)
# ---------------------------------------------------------------------------

def _n2_vertices(adj: Dict[int, Set[int]], v: int) -> Set[int]:
    """N²(v) = vertices within 2 hops of v (including v itself)."""
    vi = int(v)
    n1 = adj.get(vi, set())
    n2: Set[int] = {vi}
    n2.update(n1)
    for u in n1:
        n2.update(adj.get(u, set()))
    return n2


def _k2_induced_edge_indices(
    inc: Dict[int, List[int]],
    adj: Dict[int, Set[int]],
    edges: np.ndarray,
    v: int,
) -> np.ndarray:
    """Induced-subgraph edge indices for N²(v).

    Returns indices of edges where BOTH endpoints are in N²(v),
    i.e. the induced subgraph of the 2nd-order neighbourhood.
    This matches the paper's "edges formed by points in N²(Pj)".
    """
    n2 = _n2_vertices(adj, v)

    # Collect candidate edge indices from incidence lists of all N² vertices.
    candidates: Set[int] = set()
    for u in n2:
        candidates.update(inc.get(u, []))

    if not candidates:
        return np.empty(0, dtype=np.int64)

    idx = np.array(sorted(candidates), dtype=np.int64)
    # Keep only edges where BOTH endpoints are in N².
    mask = np.array(
        [(int(edges[i, 0]) in n2 and int(edges[i, 1]) in n2) for i in idx],
        dtype=bool,
    )
    return idx[mask]


# ---------------------------------------------------------------------------
# Step 1 – Global cut (§3.2, Eq. 6-8)
# ---------------------------------------------------------------------------

def _global_cut(graph: EdgeGraph, a: float) -> EdgeGraph:
    """Remove Global_Long_Edges.

    Eq. 6:  GCV(Pi) = μ_G + A · (μ_G / μ_L¹(Pi)) · σ_G
    where μ_G = global mean, σ_G = global std (ddof=1), μ_L¹(Pi) = local mean.

    An edge incident to Pi is removed if its length >= GCV(Pi) (Eq. 7).
    """
    if graph.empty():
        return graph

    verts = _vertices(graph)
    inc = _incidence(verts, graph.edges)

    mu_g = float(np.mean(graph.lengths))
    sigma_g = _paper_std(graph.lengths)

    remove = np.zeros(len(graph.lengths), dtype=bool)
    for v in verts:
        vi = int(v)
        idx = inc[vi]
        if not idx:
            continue
        edge_idx = np.array(idx, dtype=np.int64)
        mu_l = float(np.mean(graph.lengths[edge_idx]))
        if mu_l < 1e-12:
            continue
        gcv = mu_g + a * (mu_g / mu_l) * sigma_g
        # Eq. 7: >= (paper and MATLAB)
        remove[edge_idx[graph.lengths[edge_idx] >= gcv]] = True

    return EdgeGraph(graph.edges[~remove], graph.lengths[~remove]) if remove.any() else graph


# ---------------------------------------------------------------------------
# Step 2 – Local cut (§3.3, Eq. 4-5, 9-10)
# ---------------------------------------------------------------------------

def _local_cut(graph: EdgeGraph, b: float) -> EdgeGraph:
    """Remove Local_Long_Edges.

    Def. 5 / Eq. 4:  Local_Variation(Pi) = std of edges DIRECTLY connected
                      to Pi (K=1 edges), with ddof=1.
    Eq. 5:            Mean_Variation(G) = mean of Local_Variation(Pi) over all Pi.
    Eq. 9:            LCV(Pj) = Mean²(Pj) + β · Mean_Variation(G)
                      where Mean²(Pj) = mean length of edges in induced N²(Pj).
    Eq. 10:           edge is Local_Long_Edge if length >= LCV.
    """
    if graph.empty():
        return graph

    verts = _vertices(graph)
    adj = _adjacency(verts, graph.edges)
    inc = _incidence(verts, graph.edges)

    # Pre-compute K=2 induced edge sets for Mean² and removal tests.
    k2: Dict[int, np.ndarray] = {}
    for v in verts:
        vi = int(v)
        k2[vi] = _k2_induced_edge_indices(inc, adj, graph.edges, vi)

    # Eq. 4 (Def. 5): Local_Variation uses K=1 edges (directly connected).
    lvs = np.zeros(len(verts))
    for i, v in enumerate(verts):
        vi = int(v)
        k1_idx = inc[vi]
        if len(k1_idx) > 1:
            lvs[i] = _paper_std(graph.lengths[np.array(k1_idx, dtype=np.int64)])

    # Eq. 5: Mean_Variation = mean of all Local_Variations.
    mv = float(np.mean(lvs))

    remove = np.zeros(len(graph.lengths), dtype=bool)
    for v in verts:
        vi = int(v)
        idx = k2[vi]
        if idx.size == 0:
            continue
        mu_lk = float(np.mean(graph.lengths[idx]))     # Mean²(Pj)
        lcv = mu_lk + b * mv                           # Eq. 9
        # Eq. 10: >= (paper and MATLAB)
        remove[idx[graph.lengths[idx] >= lcv]] = True

    return EdgeGraph(graph.edges[~remove], graph.lengths[~remove]) if remove.any() else graph


# ---------------------------------------------------------------------------
# Step 3 – Chain / neck removal (§3.3, Observation 1 & Def. 10-13)
# (Paper only – not in MATLAB reference code)
# ---------------------------------------------------------------------------

def _observation1(graph: EdgeGraph) -> EdgeGraph:
    """Observation 1: delete edge (Pj, Pk) if degree(Pj)=2 and degree(Pk)=2."""
    if graph.empty():
        return graph

    verts = _vertices(graph)
    inc = _incidence(verts, graph.edges)
    deg = {int(v): len(inc[int(v)]) for v in verts}

    keep = np.array(
        [not (deg.get(int(u), 0) == 2 and deg.get(int(v), 0) == 2) for u, v in graph.edges],
        dtype=bool,
    )
    return EdgeGraph(graph.edges[keep], graph.lengths[keep]) if not keep.all() else graph


def _local_aggregation(graph: EdgeGraph, geodata: np.ndarray) -> EdgeGraph:
    """Def. 10-13: local aggregation force criterion.

    Def. 10:  F(Pj, Pk) = (1/d²) · ê  = (Pk-Pj) / d³    for Pk ∈ N²(Pj)
    Def. 11:  FC(Pj) = Σ F(Pj, Pk)                        (cohesive force)
    Def. 12:  Local_Agg_Set(Pj) = {Pk ∈ N¹(Pj) : angle(FC, F) < 90°}
    Def. 13:  Keep edge (Pj, Pk) iff Pk ∈ Agg(Pj) AND Pj ∈ Agg(Pk).
    """
    if graph.empty():
        return graph

    verts = _vertices(graph)
    if len(verts) < 3:
        return graph

    adj = _adjacency(verts, graph.edges)

    # N²(Pj) for cohesive force (excluding Pj).
    n2_ex: Dict[int, Set[int]] = {}
    for v in verts:
        vi = int(v)
        s = _n2_vertices(adj, vi)
        s.discard(vi)
        n2_ex[vi] = s

    # Cohesive force FC(Pj) (Def. 10-11).
    fc: Dict[int, np.ndarray] = {}
    for v in verts:
        vi = int(v)
        force = np.zeros(2)
        pv = geodata[vi]
        for pk in n2_ex[vi]:
            diff = geodata[pk] - pv
            d2 = float(np.dot(diff, diff))
            if d2 < 1e-24:
                continue
            force += diff / (d2 * np.sqrt(d2))      # (Pk - Pj) / d³
        fc[vi] = force

    # Local_Agg_Set (Def. 12): direct neighbours where angle(FC, F) < 90°.
    agg: Dict[int, Set[int]] = {}
    for v in verts:
        vi = int(v)
        f = fc[vi]
        if np.dot(f, f) < 1e-24:
            # Zero FC (symmetric neighbourhood) → keep all direct neighbours.
            agg[vi] = adj.get(vi, set()).copy()
        else:
            pv = geodata[vi]
            agg[vi] = {pk for pk in adj.get(vi, set()) if np.dot(f, geodata[pk] - pv) > 0}

    # Edge filtering (Def. 13): keep iff both endpoints agree.
    keep = np.array(
        [int(ev) in agg.get(int(eu), set()) and int(eu) in agg.get(int(ev), set())
         for eu, ev in graph.edges],
        dtype=bool,
    )
    return EdgeGraph(graph.edges[keep], graph.lengths[keep]) if not keep.all() else graph


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _ascdt_pipeline(
    graph: EdgeGraph,
    geodata: np.ndarray,
    a: float,
    b: float,
    minpts: int,
    enable_step3: bool,
) -> List[EdgeGraph]:
    """ASCDT Steps 1 → 2 → (optional 3) for one connected subgraph."""
    # Step 1 – Global cut
    after1 = _global_cut(graph, a)
    if not after1.empty():
        after1 = _prune_small_components(after1, minpts)
    subs1 = _split(after1) if not after1.empty() else []
    if not subs1:
        return []

    clusters: List[EdgeGraph] = []
    for sg1 in subs1:
        # Step 2 – Local cut
        after2 = _local_cut(sg1, b)
        if not after2.empty():
            after2 = _prune_small_components(after2, minpts)
        subs2 = _split(after2) if not after2.empty() else []
        if not subs2:
            continue

        if not enable_step3:
            clusters.extend(subs2)
            continue

        for sg2 in subs2:
            # Step 3 – Chain / neck removal
            sg3 = _observation1(sg2)
            if not sg3.empty():
                sg3 = _local_aggregation(sg3, geodata)
            if not sg3.empty():
                sg3 = _prune_small_components(sg3, minpts)
            subs3 = _split(sg3) if not sg3.empty() else []
            clusters.extend(subs3)

    return clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ascdt_cluster(
    geodata: np.ndarray,
    *,
    a: float = 1.0,
    b: float = 1.1,
    minpts: int = 5,
    min_cluster_size: int = 1,
    enable_step3: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """
    ASCDT clustering on 2D coordinates.

    Implementation aligned with Deng et al. (2011) paper and MATLAB reference.

    Parameters
    ----------
    geodata : (n, 2) array
        2D point coordinates.
    a : float
        Global cut parameter A (Eq. 6). Default 1.0 (paper).
    b : float
        Local cut parameter β (Eq. 9). Default 1.1 (MATLAB reference).
        Paper recommends 1.0-1.5.
    minpts : int
        Intermediate pruning threshold: connected components with <= minpts
        vertices are removed after each step (matches MATLAB MINPTS=5).
    min_cluster_size : int
        Final filter: clusters smaller than this are labelled noise (-1).
    enable_step3 : bool
        Whether to run Step 3 (Observation 1 + local aggregation forces).
        Default False (matches MATLAB reference which omits Step 3).
        Can be too aggressive on dense, overlapping data like scRNA-seq.
    random_state : int
        Seed for Delaunay jitter fallback only.

    Returns
    -------
    labels : (n,) int64 array
        Cluster IDs [0..K-1]; noise = -1.
    """
    pts = np.asarray(geodata, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("ASCDT expects shape (n, 2).")

    n = pts.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n <= 2:
        return np.zeros(n, dtype=np.int64)

    graph = _build_delaunay(pts, random_state)
    if graph.empty():
        return np.full(n, -1, dtype=np.int64)

    # Initial pruning of tiny components (matches MATLAB subgraph()).
    graph = _prune_small_components(graph, minpts)
    if graph.empty():
        return np.full(n, -1, dtype=np.int64)

    # Run pipeline on each initial connected subgraph.
    initial = _split(graph)
    all_clusters: List[EdgeGraph] = []
    for sg in initial:
        all_clusters.extend(_ascdt_pipeline(sg, pts, a, b, minpts, enable_step3))

    # Assign labels.
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for sg in all_clusters:
        if sg.empty():
            continue
        v = np.unique(sg.edges.ravel())
        if len(v) < min_cluster_size:
            continue
        labels[v] = cid
        cid += 1
    return labels
