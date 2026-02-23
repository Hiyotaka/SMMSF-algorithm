"""SMMSF clustering (Python implementation).

This module ports the original MATLAB implementation in this repository:
normalize -> alpha-MST forest construction -> density-increment split ->
two-stage merge.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


def smmsf_clustering(
    data: np.ndarray,
    k: int,
    *,
    alpha: int = 3,
    stage1_threshold: float = 1.5,
    return_details: bool = False,
):
    """Cluster data with SMMSF.

    Args:
        data: Array of shape (n_samples, n_features).
        k: Target number of clusters.
        alpha: Number of MST rounds used to build the MSF (default 3).
        stage1_threshold: Threshold in merge stage-1 condition 3.
        return_details: If True, also return intermediate graphs/clusters.

    Returns:
        labels: 1D integer labels (1-based, MATLAB-compatible).
        details: Optional dict with intermediate artifacts.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("`data` must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("SMMSF requires at least 2 samples.")
    if k < 1:
        raise ValueError("`k` must be >= 1.")

    x_norm = _minmax_normalize(x)
    mst1, mst_alpha, pairwise_dist = _build_alpha_mst_union(x_norm, alpha=alpha)
    clusters, centers, rho = _split_tree(x_norm, mst1, mst_alpha)
    labels = _two_round_merge(
        clusters=clusters,
        mst1=mst1,
        mst_alpha=mst_alpha,
        data=x_norm,
        k=k,
        cluster_centers=centers,
        stage1_threshold=stage1_threshold,
    )

    if return_details:
        details = {
            "normalized_data": x_norm,
            "pairwise_distance": pairwise_dist,
            "mst1": mst1,
            "mst_alpha": mst_alpha,
            "split_clusters": clusters,
            "split_centers": centers,
            "density_increment": rho,
        }
        return labels, details
    return labels


def _minmax_normalize(data: np.ndarray) -> np.ndarray:
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    span = data_max - data_min
    span[span == 0] = 1.0
    out = (data - data_min) / span
    out[~np.isfinite(out)] = 0.0
    return out


def _build_alpha_mst_union(data: np.ndarray, alpha: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = cdist(data, data, metric="euclidean")
    np.fill_diagonal(dist, 0.0)

    remaining = dist.copy()
    mst1 = None
    mst_union = np.zeros_like(dist)

    for round_id in range(alpha):
        tree_sparse = minimum_spanning_tree(csr_matrix(remaining))
        tree = tree_sparse.toarray()
        tree = np.maximum(tree, tree.T)

        edges = np.argwhere(np.triu(tree > 0, k=1))
        if edges.size == 0:
            break

        if round_id == 0:
            mst1 = tree.copy()

        for u, v in edges:
            w = dist[u, v]
            mst_union[u, v] = w
            mst_union[v, u] = w
            remaining[u, v] = 0.0
            remaining[v, u] = 0.0

    if mst1 is None:
        raise RuntimeError("Failed to build the first MST. Check input data connectivity.")

    return mst1, mst_union, dist


def _split_tree(
    data: np.ndarray,
    mst1: np.ndarray,
    mst_alpha: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    n = data.shape[0]

    t1 = (mst1 > 0).astype(float)
    avg_w_t1 = _safe_div(np.sum(mst1, axis=1), np.sum(t1, axis=1))
    deg_t1 = np.sum(t1, axis=1)
    rho1 = _safe_div(deg_t1, avg_w_t1)

    mst_filtered = mst_alpha.copy()
    deg_alpha = np.sum(mst_filtered > 0, axis=1)
    avg_w_alpha = _safe_div(np.sum(mst_filtered, axis=1), deg_alpha)

    # Remove overly long edges as in MATLAB: edge(i,j) is cut if it is too long
    # for either endpoint i or j.
    th_i = 2.0 * avg_w_alpha[:, None]
    th_j = 2.0 * avg_w_alpha[None, :]
    long_edge = (mst_filtered > 0) & ((mst_filtered > th_i) | (mst_filtered > th_j))
    mst_filtered[long_edge] = 0.0
    mst_filtered = np.maximum(mst_filtered, mst_filtered.T)

    t3 = (mst_filtered > 0).astype(float)
    deg_t3 = np.sum(t3, axis=1)
    avg_w_t3 = _safe_div(np.sum(mst_filtered, axis=1), deg_t3)
    rho3 = _safe_div(deg_t3, avg_w_t3)
    rho = rho3 - rho1

    rep = np.zeros(n, dtype=int)
    for i in range(n):
        neighbors = np.where(mst_filtered[i, :] > 0)[0]
        connected = np.concatenate(([i], neighbors))
        rep[i] = int(connected[np.argmax(rho[connected])])

    visited = np.zeros(n, dtype=int)
    flag = 0
    sup = 0
    for i in range(n):
        if visited[i] == 0:
            sup = i
            flag += 1
            hops = 0
            while rep[sup] != sup and hops <= n:
                visited[sup] = flag
                sup = int(rep[sup])
                hops += 1
            if hops > n:
                sup = i
        rep[visited == flag] = sup

    centers = np.where(rep == np.arange(n))[0]
    clusters = [np.where(rep == center)[0] for center in centers]
    clusters, centers = _redistribute_singletons(data, clusters, centers)
    return clusters, centers, rho


def _redistribute_singletons(
    data: np.ndarray,
    clusters: List[np.ndarray],
    centers: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    sizes = np.array([c.size for c in clusters], dtype=int)
    if not np.any(sizes == 1):
        return clusters, centers

    remove_idx = np.zeros(len(clusters), dtype=bool)
    for i, cluster in enumerate(clusters):
        if cluster.size != 1:
            continue

        singleton = data[cluster[0] : cluster[0] + 1, :]
        nearest = -1
        min_dist = np.inf

        for j, target_cluster in enumerate(clusters):
            if i == j or target_cluster.size == 0:
                continue
            d = cdist(singleton, data[target_cluster, :])
            local_min = float(np.min(d))
            if local_min < min_dist:
                min_dist = local_min
                nearest = j

        if nearest >= 0:
            clusters[nearest] = np.concatenate([clusters[nearest], cluster]).astype(int)
            clusters[i] = np.array([], dtype=int)
            remove_idx[i] = True

    clusters = [c for c in clusters if c.size > 0]
    if centers.size == remove_idx.size:
        centers = centers[~remove_idx]
    return clusters, centers


def _two_round_merge(
    clusters: List[np.ndarray],
    mst1: np.ndarray,
    mst_alpha: np.ndarray,
    data: np.ndarray,
    k: int,
    cluster_centers: np.ndarray,
    stage1_threshold: float,
) -> np.ndarray:
    n = data.shape[0]
    if not clusters:
        return np.zeros(n, dtype=int)

    clusters = _merge_stage1(
        clusters=clusters,
        mst1=mst1,
        data=data,
        cluster_centers=cluster_centers,
        k=k,
        stage1_threshold=stage1_threshold,
    )

    labels = _merge_stage2(clusters=clusters, mst_alpha=mst_alpha, n=n, k=k)
    return labels


def _merge_stage1(
    clusters: List[np.ndarray],
    mst1: np.ndarray,
    data: np.ndarray,
    cluster_centers: np.ndarray,
    k: int,
    stage1_threshold: float,
) -> List[np.ndarray]:
    num_c = len(clusters)
    if num_c <= k:
        return clusters

    threshold_n = np.sqrt(data.shape[0])
    cluster_sizes = np.array([c.size for c in clusters], dtype=float)

    center_data = data[cluster_centers, :] if cluster_centers.size else np.zeros((num_c, data.shape[1]))
    center_dis = cdist(center_data, center_data) if center_data.size else np.zeros((num_c, num_c))

    conn = _get_connectionship(clusters, mst1)
    max_dis = _get_max_cluster_distance(center_data, data, clusters)
    merge_filter = _get_merge_filter(max_dis, center_dis, stage1_threshold)

    visited = np.zeros(num_c, dtype=int)
    for i in range(num_c - 1):
        current_label = int(np.max(visited)) + 1
        if visited[i] > 0:
            current_label = int(visited[i])
        else:
            visited[i] = current_label

        for j in range(i + 1, num_c):
            cond = (
                conn[i, j] > 0
                and merge_filter[i, j] > 0
                and visited[j] == 0
                and cluster_sizes[i] < threshold_n
                and cluster_sizes[j] < threshold_n
            )
            if cond:
                visited[j] = current_label

    if visited[-1] == 0:
        visited[-1] = int(np.max(visited)) + 1

    merged_clusters: List[np.ndarray] = []
    for label in np.unique(visited):
        idx = np.where(visited == label)[0]
        merged = np.concatenate([clusters[t] for t in idx]).astype(int)
        merged_clusters.append(merged)

    return merged_clusters


def _merge_stage2(clusters: List[np.ndarray], mst_alpha: np.ndarray, n: int, k: int) -> np.ndarray:
    num_c = len(clusters)
    neighbors, ic_values, heap = _initialize_priority_structures(clusters, mst_alpha)

    active = np.ones(num_c, dtype=bool)
    active_count = int(np.sum(active))

    while active_count > k:
        selected_from_heap = True
        if not heap:
            active_idx = np.where(active)[0]
            if active_idx.size < 2:
                break
            i, j = sorted(active_idx[:2])
            candidate_ic = ic_values[i, j]
            selected_from_heap = False
        else:
            neg_ic, i, j = heapq.heappop(heap)
            candidate_ic = -neg_ic

        if i >= num_c or j >= num_c or (not active[i]) or (not active[j]) or i == j:
            continue

        if selected_from_heap:
            if j not in neighbors[i]:
                continue
        else:
            neighbors[i].add(j)
            neighbors[j].add(i)

        current_ic = ic_values[i, j]
        if not np.isfinite(current_ic):
            continue

        if selected_from_heap:
            tolerance = max(1.0, abs(current_ic)) * np.finfo(float).eps
            if abs(current_ic - candidate_ic) > tolerance:
                continue

        clusters[i] = np.concatenate([clusters[i], clusters[j]]).astype(int)
        clusters[j] = np.array([], dtype=int)
        active[j] = False
        active_count -= 1

        ic_values[j, :] = 0.0
        ic_values[:, j] = 0.0

        candidate_neighbors = (neighbors[i] | neighbors[j]) - {i, j}
        neighbors[j].clear()
        neighbors[i].clear()

        for m in list(candidate_neighbors):
            if m >= num_c:
                continue

            neighbors[m].discard(j)
            if not active[m]:
                continue

            edge_block = mst_alpha[np.ix_(clusters[i], clusters[m])]
            if np.any(edge_block > 0):
                vert_ratio, d_val = _compute_vert_ratio(clusters[i], clusters[m], mst_alpha)
                new_ic = vert_ratio * d_val
                if not np.isfinite(new_ic):
                    new_ic = 0.0

                neighbors[i].add(m)
                neighbors[m].add(i)
                ic_values[i, m] = new_ic
                ic_values[m, i] = new_ic

                a, b = sorted((i, m))
                heapq.heappush(heap, (-new_ic, a, b))
            else:
                ic_values[i, m] = 0.0
                ic_values[m, i] = 0.0
                neighbors[m].discard(i)

    final_clusters = [clusters[idx] for idx in range(num_c) if active[idx] and clusters[idx].size > 0]

    labels = np.zeros(n, dtype=int)
    if not final_clusters:
        return labels

    kept = min(k, len(final_clusters))
    for label_id, cluster in enumerate(final_clusters[:kept], start=1):
        labels[cluster] = label_id

    if len(final_clusters) > kept:
        labels[np.concatenate(final_clusters[kept:])] = kept

    return labels


def _initialize_priority_structures(
    clusters: List[np.ndarray],
    mst_alpha: np.ndarray,
) -> Tuple[List[set], np.ndarray, List[Tuple[float, int, int]]]:
    num_c = len(clusters)
    neighbors = [set() for _ in range(num_c)]
    ic_values = np.zeros((num_c, num_c), dtype=float)
    heap: List[Tuple[float, int, int]] = []

    for i in range(num_c - 1):
        for j in range(i + 1, num_c):
            edge_block = mst_alpha[np.ix_(clusters[i], clusters[j])]
            if np.any(edge_block > 0):
                vert_ratio, d_val = _compute_vert_ratio(clusters[i], clusters[j], mst_alpha)
                value = vert_ratio * d_val
                if not np.isfinite(value):
                    value = 0.0

                neighbors[i].add(j)
                neighbors[j].add(i)
                ic_values[i, j] = value
                ic_values[j, i] = value
                heapq.heappush(heap, (-value, i, j))

    return neighbors, ic_values, heap


def _compute_vert_ratio(c1: np.ndarray, c2: np.ndarray, mst_alpha: np.ndarray) -> Tuple[float, float]:
    edge = mst_alpha[np.ix_(c1, c2)]
    rows, cols = np.where(edge > 0)
    if rows.size == 0:
        return 0.0, 0.0

    num1 = c1.size
    num2 = c2.size
    v1 = np.unique(rows)
    v2 = np.unique(cols)

    vert_ratio = float((v1.size + v2.size) / (num1 + num2))
    avg_edge = float(np.mean(edge[rows, cols]))

    v1_pts = c1[v1]
    v2_pts = c2[v2]
    rest1 = np.setdiff1d(c1, v1_pts, assume_unique=False)
    rest2 = np.setdiff1d(c2, v2_pts, assume_unique=False)

    conn1 = mst_alpha[np.ix_(v1_pts, rest1)] if rest1.size else np.empty((v1_pts.size, 0))
    conn2 = mst_alpha[np.ix_(v2_pts, rest2)] if rest2.size else np.empty((v2_pts.size, 0))

    avg_conn1 = _positive_mean(conn1)
    avg_conn2 = _positive_mean(conn2)
    d1 = _symmetric_ratio(avg_edge, avg_conn1)
    d2 = _symmetric_ratio(avg_edge, avg_conn2)

    d = min(d1, d2)
    if not np.isfinite(d):
        d = 0.0
    return vert_ratio, d


def _get_connectionship(clusters: List[np.ndarray], mst1: np.ndarray) -> np.ndarray:
    n = len(clusters)
    connection = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        for j in range(i + 1, n):
            block = mst1[np.ix_(clusters[i], clusters[j])]
            if np.any(block > 0):
                connection[i, j] = 1
    return connection


def _get_max_cluster_distance(
    center_data: np.ndarray,
    data: np.ndarray,
    clusters: List[np.ndarray],
) -> np.ndarray:
    max_dist = np.zeros((len(clusters),), dtype=float)
    for i, cluster in enumerate(clusters):
        if cluster.size == 0:
            continue
        d = cdist(data[cluster], center_data[i : i + 1])
        max_dist[i] = float(np.max(d)) if d.size else 0.0
    return max_dist


def _get_merge_filter(max_dis: np.ndarray, center_dis: np.ndarray, threshold: float) -> np.ndarray:
    n = max_dis.size
    merge_filter = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if max_dis[i] + max_dis[j] >= threshold * center_dis[i, j]:
                merge_filter[i, j] = 1
    return merge_filter


def _safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.zeros_like(numer, dtype=float)
    mask = denom > np.finfo(float).eps
    out[mask] = numer[mask] / denom[mask]
    return out


def _positive_mean(arr: np.ndarray) -> float:
    vals = arr[arr > 0]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def _symmetric_ratio(a: float, b: float) -> float:
    if (not np.isfinite(a)) or (not np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return float(min(a / b, b / a))


__all__ = ["smmsf_clustering"]
