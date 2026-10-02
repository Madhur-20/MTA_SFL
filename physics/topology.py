"""MTA-SFL Network Topology — Directed Graph, Clustering Coefficient & Queues.

Constructs the directed adjacency matrix from SINR measurements, computes the
local clustering coefficient (Eq. 9), tracks per-node queue congestion, and
provides multi-hop cloud-feasibility queries.

References
----------
  Eq. 9:  CC_u(t) = |E_u(t)| / [ k_u(t) · (k_u(t) − 1) ]   if k_u ≥ 2
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Set

import numpy as np

from mta_sfl.config.system_config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Directed Adjacency Matrix
# ──────────────────────────────────────────────────────────────────────────────
def build_adjacency_matrix(
    sinr_matrix: np.ndarray,
    gamma_th: float,
) -> np.ndarray:
    """Build a directed adjacency matrix from an SINR matrix.

    A link from *m* to *n* exists iff SINR_{m,n} ≥ γ_th.

    Args:
        sinr_matrix: N×N array of linear-scale SINR values.
        gamma_th: Decoding threshold in linear scale.

    Returns:
        N×N integer adjacency matrix (0 / 1).  Self-loops are always 0.
    """
    adj = (sinr_matrix >= gamma_th).astype(np.int32)
    np.fill_diagonal(adj, 0)
    return adj


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Local Clustering Coefficient  (Eq. 9)
# ──────────────────────────────────────────────────────────────────────────────
def local_clustering_coefficient(
    adj_matrix: np.ndarray,
    node_idx: int,
) -> float:
    """Local clustering coefficient for a single node  (Eq. 9).

    CC_u = |E_u| / ( k_u · (k_u − 1) )     if k_u ≥ 2
    CC_u = 0                                 otherwise

    where:
    * N_u = {v : A[u, v] = 1}         (outgoing neighbours of u)
    * k_u = |N_u|                      (out-degree)
    * E_u = {(v, w) : v, w ∈ N_u, A[v, w] = 1}  (directed edges among neighbours)

    Args:
        adj_matrix: N×N integer adjacency matrix.
        node_idx: Index of the target node.

    Returns:
        Clustering coefficient in [0, 1].
    """
    neighbours = np.where(adj_matrix[node_idx] == 1)[0]
    k_u: int = len(neighbours)
    if k_u < 2:
        return 0.0

    # Sub-matrix of directed edges among the neighbours
    sub = adj_matrix[np.ix_(neighbours, neighbours)]
    # Exclude self-loops (should already be 0, but guard)
    np.fill_diagonal(sub, 0)
    e_u: int = int(np.sum(sub))

    return e_u / (k_u * (k_u - 1))


def all_clustering_coefficients(adj_matrix: np.ndarray) -> np.ndarray:
    """Clustering coefficients for every node.

    Args:
        adj_matrix: N×N adjacency matrix.

    Returns:
        1-D array of length N with CC values.
    """
    N = adj_matrix.shape[0]
    return np.array(
        [local_clustering_coefficient(adj_matrix, i) for i in range(N)]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Queue Tracker
# ──────────────────────────────────────────────────────────────────────────────
class QueueTracker:
    """Per-node payload / packet queue tracker.

    Maintains Q_{load,u}(t) in bits for each node.  Supports enqueue,
    dequeue, query, and reset operations.

    Args:
        num_nodes: Total number of nodes to track.
    """

    def __init__(self, num_nodes: int) -> None:
        self._queues: np.ndarray = np.zeros(num_nodes, dtype=np.float64)

    # --------------------------------------------------------- enqueue
    def enqueue(self, node_idx: int, bits: float) -> None:
        """Add *bits* to the node's queue."""
        self._queues[node_idx] += bits

    # --------------------------------------------------------- dequeue
    def dequeue(self, node_idx: int, bits: float) -> float:
        """Remove up to *bits* from the queue.

        Returns:
            Actually removed amount (may be less than requested).
        """
        available = self._queues[node_idx]
        removed = min(available, bits)
        self._queues[node_idx] -= removed
        return float(removed)

    # --------------------------------------------------------- queries
    def get_load(self, node_idx: int) -> float:
        """Current queue load in bits."""
        return float(self._queues[node_idx])

    def get_all_loads(self) -> np.ndarray:
        """Copy of all queue loads."""
        return self._queues.copy()

    # --------------------------------------------------------- reset
    def reset(self, node_idx: Optional[int] = None) -> None:
        """Reset a single node's queue, or all queues if *node_idx* is None."""
        if node_idx is not None:
            self._queues[node_idx] = 0.0
        else:
            self._queues[:] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Topology Manager
# ──────────────────────────────────────────────────────────────────────────────
class TopologyManager:
    """High-level manager combining adjacency, clustering, and queue state.

    Provides convenience methods for neighbour queries, relay-candidate
    selection, and multi-hop cloud-reachability checks.

    Args:
        cfg: Simulation configuration (uses ``cfg.channel.sinr_threshold_linear``).
        num_nodes: Total number of nodes in the network.  If *None*, derived
            from ``cfg.num_iot + cfg.num_uav + cfg.num_cloud``.
    """

    def __init__(
        self,
        cfg: SimulationConfig,
        num_nodes: Optional[int] = None,
    ) -> None:
        self._cfg = cfg
        n = num_nodes if num_nodes is not None else (
            cfg.num_iot + cfg.num_uav + cfg.num_cloud
        )
        self._adjacency: np.ndarray = np.zeros((n, n), dtype=np.int32)
        self._queue: QueueTracker = QueueTracker(n)

    # --------------------------------------------------------- update
    def update(self, sinr_matrix: np.ndarray) -> None:
        """Rebuild the adjacency matrix from a fresh SINR snapshot.

        Args:
            sinr_matrix: N×N linear-scale SINR matrix.
        """
        gamma_th = self._cfg.channel.sinr_threshold_linear
        self._adjacency = build_adjacency_matrix(sinr_matrix, gamma_th)

    # --------------------------------------------------------- neighbours
    def get_neighbors(self, node_idx: int) -> List[int]:
        """Outgoing neighbours of *node_idx*."""
        return list(np.where(self._adjacency[node_idx] == 1)[0])

    def get_bidirectional_neighbors(self, node_idx: int) -> List[int]:
        """Nodes reachable in **both** directions (i→j ∧ j→i)."""
        out = set(np.where(self._adjacency[node_idx] == 1)[0])
        inc = set(np.where(self._adjacency[:, node_idx] == 1)[0])
        return sorted(out & inc)

    # --------------------------------------------------------- clustering
    def get_clustering_coefficient(self, node_idx: int) -> float:
        """Local CC for one node  (Eq. 9)."""
        return local_clustering_coefficient(self._adjacency, node_idx)

    # --------------------------------------------------------- relay / cloud
    def get_relay_candidates(
        self,
        primary_uav_idx: int,
        cloud_reachable: Set[int],
    ) -> List[int]:
        """Return N_j(t) ∩ U_{cloud}(t).

        Args:
            primary_uav_idx: Index of the querying primary UAV.
            cloud_reachable: Set of node indices that can reach the Cloud.

        Returns:
            Sorted list of neighbour indices that are also cloud-reachable.
        """
        neighbours = set(self.get_neighbors(primary_uav_idx))
        return sorted(neighbours & cloud_reachable)

    def has_cloud_path(
        self,
        node_idx: int,
        cloud_indices: Set[int],
    ) -> bool:
        """BFS check whether *node_idx* can reach any Cloud gateway.

        Args:
            node_idx: Starting node.
            cloud_indices: Set of Cloud gateway indices.

        Returns:
            True if a directed path exists from *node_idx* to any cloud node.
        """
        visited: Set[int] = set()
        frontier: collections.deque[int] = collections.deque([node_idx])

        while frontier:
            cur = frontier.popleft()
            if cur in cloud_indices:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in self.get_neighbors(cur):
                if nxt not in visited:
                    frontier.append(nxt)

        return False

    def cloud_reachable_set(self, cloud_indices: Set[int]) -> Set[int]:
        """Return the set of *all* nodes that can reach any Cloud gateway.

        Performs BFS from every node; suitable for moderate-size networks.

        Args:
            cloud_indices: Set of Cloud gateway indices.

        Returns:
            Set of node indices with a directed path to a cloud node.
        """
        N = self._adjacency.shape[0]
        reachable: Set[int] = set()
        for n in range(N):
            if self.has_cloud_path(n, cloud_indices):
                reachable.add(n)
        return reachable

    # --------------------------------------------------------- properties
    @property
    def adjacency(self) -> np.ndarray:
        """Current directed adjacency matrix (read-only view)."""
        return self._adjacency

    @property
    def queue(self) -> QueueTracker:
        """Per-node queue tracker."""
        return self._queue


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running topology.py self-tests …")

    # ----- 1. Adjacency matrix thresholding -----
    sinr = np.array([
        [10.0, 5.0,  0.1],
        [2.0,  10.0, 3.0],
        [0.5,  4.0,  10.0],
    ])
    gamma_th = 2.5
    adj = build_adjacency_matrix(sinr, gamma_th)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    assert np.array_equal(adj, expected), f"Adjacency mismatch:\n{adj}"
    print("  [OK] Adjacency matrix thresholding")

    # ----- 2. CC = 1.0 for fully-connected neighbours -----
    # Node 0 → {1, 2, 3}, and 1↔2↔3 fully connected (6 directed edges)
    adj4 = np.zeros((4, 4), dtype=np.int32)
    adj4[0, 1] = adj4[0, 2] = adj4[0, 3] = 1
    adj4[1, 2] = adj4[1, 3] = 1
    adj4[2, 1] = adj4[2, 3] = 1
    adj4[3, 1] = adj4[3, 2] = 1
    cc0 = local_clustering_coefficient(adj4, 0)
    assert abs(cc0 - 1.0) < 1e-9, f"CC(0) = {cc0}, expected 1.0"
    print("  [OK] CC = 1.0 for full 3-clique neighbours")

    # ----- 3. CC = 0 for k < 2 -----
    adj2 = np.zeros((3, 3), dtype=np.int32)
    adj2[0, 1] = 1
    cc_single = local_clustering_coefficient(adj2, 0)
    assert cc_single == 0.0, f"CC = {cc_single}, expected 0.0"
    print("  [OK] CC = 0 for single neighbour")

    # ----- 4. CC for a known small graph -----
    # Node 0 → {1, 2}; 1 → {2}; 2 → ∅
    # E_0 = { (1, 2) }  ⇒ |E_0| = 1
    # CC_0 = 1 / (2 · 1) = 0.5
    adj_known = np.zeros((3, 3), dtype=np.int32)
    adj_known[0, 1] = adj_known[0, 2] = 1
    adj_known[1, 2] = 1
    cc_known = local_clustering_coefficient(adj_known, 0)
    assert abs(cc_known - 0.5) < 1e-9, f"CC = {cc_known}, expected 0.5"
    print("  [OK] CC = 0.5 for known directed triangle")

    # ----- 5. QueueTracker -----
    qt = QueueTracker(3)
    qt.enqueue(0, 1000.0)
    qt.enqueue(0, 500.0)
    assert qt.get_load(0) == 1500.0
    assert qt.get_load(1) == 0.0
    removed = qt.dequeue(0, 800.0)
    assert removed == 800.0
    assert qt.get_load(0) == 700.0
    removed2 = qt.dequeue(0, 9999.0)
    assert removed2 == 700.0
    assert qt.get_load(0) == 0.0
    qt.enqueue(2, 42.0)
    loads = qt.get_all_loads()
    assert loads[2] == 42.0
    qt.reset()
    assert np.all(qt.get_all_loads() == 0.0)
    print("  [OK] QueueTracker enqueue / dequeue / reset")

    # ----- 6. TopologyManager -----
    cfg = SimulationConfig(num_iot=0, num_uav=4, num_cloud=0, seed=1)
    # Override threshold for test
    cfg.channel.sinr_threshold_dB = 3.0  # ~2.0 linear

    tm = TopologyManager(cfg, num_nodes=4)
    sinr4 = np.array([
        [10, 5, 1, 1],
        [1, 10, 5, 1],
        [1, 1, 10, 5],
        [5, 1, 1, 10],
    ], dtype=np.float64)
    tm.update(sinr4)

    # γ_th ≈ 2.0 → thresholding:
    #  row0: [0,1,0,0]  row1: [0,0,1,0]  row2: [0,0,0,1]  row3: [1,0,0,0]
    assert tm.get_neighbors(0) == [1]
    assert tm.get_neighbors(1) == [2]
    assert tm.get_neighbors(2) == [3]
    assert tm.get_neighbors(3) == [0]
    print("  [OK] TopologyManager.get_neighbors")

    # Bidirectional (ring graph — no bidirectional links)
    assert tm.get_bidirectional_neighbors(0) == []
    print("  [OK] TopologyManager.get_bidirectional_neighbors (ring)")

    # Relay candidates
    assert tm.get_relay_candidates(0, cloud_reachable={1, 3}) == [1]
    assert tm.get_relay_candidates(0, cloud_reachable={2, 3}) == []
    print("  [OK] TopologyManager.get_relay_candidates")

    # BFS cloud path: 0 → 1 → 2 → 3 (ring)
    assert tm.has_cloud_path(0, cloud_indices={2})   # 0→1→2
    assert tm.has_cloud_path(0, cloud_indices={3})   # 0→1→2→3
    assert tm.has_cloud_path(1, cloud_indices={0})   # 1→2→3→0 (ring)

    # Test unreachable: break the ring by building a non-cyclic graph
    sinr_broken = np.array([
        [10, 5, 0, 0],
        [0, 10, 5, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 10],
    ], dtype=np.float64)
    tm2 = TopologyManager(cfg, num_nodes=4)
    tm2.update(sinr_broken)
    # 0→1→2, node 3 is isolated
    assert not tm2.has_cloud_path(0, cloud_indices={3})  # no path to 3
    assert tm2.has_cloud_path(0, cloud_indices={2})       # 0→1→2
    print("  [OK] TopologyManager.has_cloud_path (BFS)")

    # Cloud reachable set (all nodes can reach all others in a ring)
    reachable = tm.cloud_reachable_set(cloud_indices={3})
    assert reachable == {0, 1, 2, 3}, f"Reachable = {reachable}"
    print("  [OK] TopologyManager.cloud_reachable_set")

    print("[PASS] topology.py — All assertions passed.")
