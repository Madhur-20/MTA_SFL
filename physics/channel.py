"""MTA-SFL Channel Model — OFDMA, Path Loss, Shadowing & True SINR (Eq. 5).

Implements the *exact* SINR formulation from the MTA-SFL paper:

    SINR_{m,n}(t) =       P_m · h_{m,n}(t)
                    ─────────────────────────────────────
                    N_0 · W_n(t) + Σ_{k∈K_int} P_k · h_{k,n}(t)

No SNR shortcuts.  Inter-cluster sub-channel collision is modelled explicitly.

References
----------
  Eq. 5:  SINR definition.
  Eq. 6:  Uplink Shannon rate  R^{UL}_{m,n} = B · log₂(1 + SINR).
  Eq. 7:  Downlink Shannon rate (symmetric form).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mta_sfl.config.system_config import (
    ChannelConfig,
    PhysicalConstants,
    PowerConfig,
    SimulationConfig,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Path-Loss Functions
# ──────────────────────────────────────────────────────────────────────────────
def path_loss_free_space_dB(
    d_m: float,
    f_hz: float,
    c: float = 299_792_458.0,
) -> float:
    """Free-space path loss (Friis) in dB.

    PL(dB) = 20 · log10(4π · d · f / c)

    Args:
        d_m: Link distance (m).  Returns 0.0 dB for d ≤ 0.
        f_hz: Carrier frequency (Hz).
        c: Speed of light (m/s).

    Returns:
        Path loss in dB (always ≥ 0).
    """
    if d_m <= 0.0:
        return 0.0
    return 20.0 * math.log10(4.0 * math.pi * d_m * f_hz / c)


def path_loss_generic_dB(
    d_m: float,
    f_hz: float,
    alpha: float,
    c: float = 299_792_458.0,
) -> float:
    """Distance-dependent path loss with arbitrary exponent α.

    PL(dB) = 10 · α · log10(d)  +  20 · log10(4π · f / c)

    For d ≤ 1 m the reference distance is clamped at 1 m (near-field guard).

    Args:
        d_m: Link distance (m).
        f_hz: Carrier frequency (Hz).
        alpha: Path-loss exponent (≥ 2).
        c: Speed of light (m/s).

    Returns:
        Path loss in dB.
    """
    d_eff = max(d_m, 1.0)
    return (
        10.0 * alpha * math.log10(d_eff)
        + 20.0 * math.log10(4.0 * math.pi * f_hz / c)
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Channel Gain (linear)
# ──────────────────────────────────────────────────────────────────────────────
def channel_gain(
    d_m: float,
    f_hz: float,
    alpha: float,
    shadow_dB: float = 0.0,
    c: float = 299_792_458.0,
) -> float:
    """Linear channel gain incorporating path loss and log-normal shadowing.

    h = 10^{ −(PL_dB + shadow_dB) / 10 }

    The shadow sample may be positive (constructive) or negative (destructive).

    Args:
        d_m: Link distance (m).
        f_hz: Carrier frequency (Hz).
        alpha: Path-loss exponent.
        shadow_dB: Pre-drawn shadowing realisation in dB (N(0, σ²)).
        c: Speed of light (m/s).

    Returns:
        Linear channel gain (dimensionless, 0 < h ≤ 1 for d ≥ 1 m).
    """
    pl_dB = path_loss_generic_dB(d_m, f_hz, alpha, c)
    total_dB = pl_dB + shadow_dB
    return 10.0 ** (-total_dB / 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  OFDMA Sub-channel Bandwidth  (Eq. 6 denominator)
# ──────────────────────────────────────────────────────────────────────────────
def ofdma_subchannel_bandwidth(W_total_hz: float, K_cluster: int) -> float:
    """OFDMA sub-channel bandwidth per admitted device.

    B_{i,j}^{UL} = W^{UL} / K_j(t)

    Intra-cluster orthogonality is strict: each device gets a disjoint
    sub-carrier partition, eliminating intra-cluster interference.

    Args:
        W_total_hz: Total cluster bandwidth (Hz).
        K_cluster: Number of admitted devices in this cluster (≥ 1).

    Returns:
        Per-device bandwidth (Hz).

    Raises:
        ValueError: If K_cluster < 1.
    """
    if K_cluster < 1:
        raise ValueError(f"K_cluster must be ≥ 1, got {K_cluster}")
    return W_total_hz / K_cluster


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SINR  (Exact Eq. 5 — No SNR Shortcuts)
# ──────────────────────────────────────────────────────────────────────────────
def compute_sinr(
    tx_idx: int,
    rx_idx: int,
    tx_power_W: float,
    bandwidth_hz: float,
    gain_matrix: np.ndarray,
    interferer_indices: List[int],
    interferer_powers: List[float],
    N0_W_per_Hz: float,
) -> float:
    """Exact SINR at receiver *rx_idx* from transmitter *tx_idx*  (Eq. 5).

        SINR = P_tx · h[tx, rx]  /  (N0 · B + Σ_k P_k · h[k, rx])

    Args:
        tx_idx: Index of the desired transmitter in the gain matrix.
        rx_idx: Index of the receiver.
        tx_power_W: Transmit power of the desired transmitter (W).
        bandwidth_hz: Sub-channel bandwidth assigned to this link (Hz).
        gain_matrix: 2-D array of linear channel gains, shape ``(N, N)``.
        interferer_indices: List of indices of co-channel interferers.
        interferer_powers: Corresponding transmit powers (W), same order.
        N0_W_per_Hz: Thermal noise PSD (W / Hz).

    Returns:
        SINR in linear scale (dimensionless).
    """
    desired_power: float = tx_power_W * float(gain_matrix[tx_idx, rx_idx])
    noise_power: float = N0_W_per_Hz * bandwidth_hz

    interference: float = 0.0
    for k_idx, p_k in zip(interferer_indices, interferer_powers):
        interference += p_k * float(gain_matrix[k_idx, rx_idx])

    denominator: float = noise_power + interference
    if denominator <= 0.0:
        return float("inf")

    return desired_power / denominator


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Shannon Rate  (Eqs. 6 & 7)
# ──────────────────────────────────────────────────────────────────────────────
def shannon_rate(bandwidth_hz: float, sinr_linear: float) -> float:
    """Directional Shannon transmission rate.

    R = B · log₂(1 + SINR)

    Args:
        bandwidth_hz: Sub-channel bandwidth (Hz).
        sinr_linear: SINR in linear scale.

    Returns:
        Achievable rate in bits / second.
    """
    if sinr_linear <= 0.0:
        return 0.0
    return bandwidth_hz * math.log2(1.0 + sinr_linear)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Interference Impact Telemetry
# ──────────────────────────────────────────────────────────────────────────────
def interference_impact_telemetry(
    tx_idx: int,
    rx_idx: int,
    tx_power_W: float,
    bandwidth_hz: float,
    gain_matrix: np.ndarray,
    interferer_indices: List[int],
    interferer_powers: List[float],
    N0_W_per_Hz: float,
) -> Dict[str, float]:
    """Compare actual SINR capacity against ideal interference-free SNR.

    Returns:
        Dictionary with keys:
        - ``sinr_dB``:  Actual SINR (dB).
        - ``snr_dB``:   Ideal interference-free SNR (dB).
        - ``capacity_actual_bps``:  Shannon rate with interference (bps).
        - ``capacity_ideal_bps``:   Shannon rate without interference (bps).
        - ``degradation_ratio``:    capacity_actual / capacity_ideal (0–1).
        - ``interference_power_W``: Total interference power at receiver (W).
    """
    # Actual SINR
    sinr_lin = compute_sinr(
        tx_idx, rx_idx, tx_power_W, bandwidth_hz,
        gain_matrix, interferer_indices, interferer_powers, N0_W_per_Hz,
    )

    # Ideal SNR (no interference)
    snr_lin = compute_sinr(
        tx_idx, rx_idx, tx_power_W, bandwidth_hz,
        gain_matrix, [], [], N0_W_per_Hz,
    )

    cap_actual = shannon_rate(bandwidth_hz, sinr_lin)
    cap_ideal = shannon_rate(bandwidth_hz, snr_lin)

    # Interference power
    interference_W: float = sum(
        p_k * float(gain_matrix[k_idx, rx_idx])
        for k_idx, p_k in zip(interferer_indices, interferer_powers)
    )

    degradation = cap_actual / cap_ideal if cap_ideal > 0.0 else 1.0

    def _lin_to_dB(x: float) -> float:
        return 10.0 * math.log10(x) if x > 0.0 else -float("inf")

    return {
        "sinr_dB": _lin_to_dB(sinr_lin),
        "snr_dB": _lin_to_dB(snr_lin),
        "capacity_actual_bps": cap_actual,
        "capacity_ideal_bps": cap_ideal,
        "degradation_ratio": degradation,
        "interference_power_W": interference_W,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Channel State Manager
# ──────────────────────────────────────────────────────────────────────────────
class ChannelState:
    """Snapshot of pairwise channel gains for all entities.

    On construction (or call to :meth:`update`) the class:

    1. Computes pairwise Euclidean distances.
    2. Selects the carrier frequency and path-loss exponent for each link
       based on the node-type pair (IoT↔UAV ⇒ 2.4 GHz / α_los;
       UAV↔UAV or UAV↔Cloud ⇒ 5.8 GHz / α_los).
    3. Draws i.i.d. log-normal shadowing samples per link.
    4. Populates the ``gain_matrix`` of linear channel gains.

    Attributes:
        gain_matrix: 2-D ``np.ndarray`` of shape ``(N, N)`` with linear
            channel gains h_{m,n}.
        distance_matrix: Pairwise distances (m).
        shadow_matrix: Shadowing realisations (dB).
    """

    def __init__(
        self,
        positions: np.ndarray,
        node_types: List[str],
        cfg: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            positions: Array of shape ``(N, 3)`` with entity coordinates.
            node_types: List of length N with type strings.
            cfg: Simulation configuration.
            rng: Seeded random generator for shadowing.
        """
        self._cfg = cfg
        self._rng = rng
        self._node_types = list(node_types)
        self._N = len(node_types)

        # Pre-compute per-link parameters (carrier freq, alpha)
        self._f_hz_matrix = np.zeros((self._N, self._N), dtype=np.float64)
        self._alpha_matrix = np.full(
            (self._N, self._N), cfg.channel.alpha_los, dtype=np.float64
        )

        ch = cfg.channel
        for i in range(self._N):
            for j in range(self._N):
                if i == j:
                    continue
                ti, tj = node_types[i], node_types[j]
                if (ti == "iot" and tj == "uav") or (ti == "uav" and tj == "iot"):
                    self._f_hz_matrix[i, j] = ch.f_c_iot_uav_hz
                elif (ti == "uav" and tj == "uav"):
                    self._f_hz_matrix[i, j] = ch.f_c_uav_uav_hz
                elif (ti == "uav" and tj == "cloud") or (ti == "cloud" and tj == "uav"):
                    self._f_hz_matrix[i, j] = ch.f_c_uav_cloud_hz
                elif (ti == "iot" and tj == "cloud") or (ti == "cloud" and tj == "iot"):
                    # IoT ↔ Cloud direct (unusual, but support it)
                    self._f_hz_matrix[i, j] = ch.f_c_iot_uav_hz
                elif ti == "cloud" and tj == "cloud":
                    self._f_hz_matrix[i, j] = ch.f_c_uav_cloud_hz
                elif ti == "iot" and tj == "iot":
                    self._f_hz_matrix[i, j] = ch.f_c_iot_uav_hz
                else:
                    self._f_hz_matrix[i, j] = ch.f_c_iot_uav_hz

                # Use NLoS exponent for IoT-IoT (ground-to-ground)
                if ti == "iot" and tj == "iot":
                    self._alpha_matrix[i, j] = ch.alpha_nlos

        # Allocate output arrays
        self.distance_matrix: np.ndarray = np.zeros(
            (self._N, self._N), dtype=np.float64
        )
        self.shadow_matrix: np.ndarray = np.zeros(
            (self._N, self._N), dtype=np.float64
        )
        self.gain_matrix: np.ndarray = np.zeros(
            (self._N, self._N), dtype=np.float64
        )

        # Initial computation
        self.update(positions)

    # ---------------------------------------------------------------- update
    def update(self, positions: np.ndarray) -> None:
        """Recompute all channel gains for new positions.

        Shadowing samples are redrawn every call (block-fading model).

        Args:
            positions: Array of shape ``(N, 3)``.
        """
        N = self._N
        sigma = self._cfg.channel.shadow_sigma_dB
        c = self._cfg.physics.c

        # Pairwise distances via broadcasting
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))         # (N, N)

        # Draw shadowing (symmetric for physical consistency)
        raw_shadow = self._rng.normal(0.0, sigma, size=(N, N))
        self.shadow_matrix = (raw_shadow + raw_shadow.T) / 2.0  # symmetrise
        np.fill_diagonal(self.shadow_matrix, 0.0)

        # Compute gains element-wise
        for i in range(N):
            for j in range(N):
                if i == j:
                    self.gain_matrix[i, j] = 0.0
                    continue
                self.gain_matrix[i, j] = channel_gain(
                    d_m=self.distance_matrix[i, j],
                    f_hz=self._f_hz_matrix[i, j],
                    alpha=self._alpha_matrix[i, j],
                    shadow_dB=self.shadow_matrix[i, j],
                    c=c,
                )


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Batch SINR (with OFDMA-aware inter-cluster interference)
# ──────────────────────────────────────────────────────────────────────────────
def compute_all_sinr(
    channel_state: ChannelState,
    cluster_assignments: Dict[int, List[int]],
    tx_powers: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute the full N×N SINR matrix with OFDMA-aware interference.

    Within a cluster, OFDMA guarantees orthogonality (no intra-cluster
    interference).  Across clusters, IoT devices that happen to occupy the
    same sub-carrier indices create inter-cluster interference.

    Sub-carrier collision model:  Cluster *j* with *K_j* admitted devices
    partitions the uplink band into *K_j* equal sub-channels.  If cluster
    *j₁* and *j₂* have devices on overlapping sub-channels (same fractional
    index), those devices are mutual interferers.

    For UAV↔UAV and UAV↔Cloud links the full backhaul/cloud bandwidth is
    used and all other concurrent transmitters on that band are interferers.

    Args:
        channel_state: Pre-computed channel gains.
        cluster_assignments: Maps primary-UAV index → list of IoT indices.
        tx_powers: 1-D array of transmit powers (W) per node, length N.
        cfg: Simulation configuration.

    Returns:
        N×N array of SINR values (linear).  Entry ``[m, n]`` is the SINR
        of a transmission from *m* received at *n*.  Entries for impossible
        links (self-loops, wrong-type) are set to 0.0.
    """
    N = channel_state.gain_matrix.shape[0]
    sinr_matrix = np.zeros((N, N), dtype=np.float64)
    N0 = cfg.physics.N0_W_per_Hz
    ch = cfg.channel

    # --- Build per-sub-channel IoT interferer map ---
    # For each cluster, assign sub-channel indices 0..K_j-1 to its IoTs.
    # Devices in *different* clusters sharing the same sub-channel index
    # and overlapping fractional bands are interferers.
    # Simplified collision model: sub-channel *s* in cluster *j* collides
    # with sub-channel *s'* in cluster *j'* iff s/K_j ranges overlap with
    # s'/K_j'.  Under equal partitioning, a device at sub-channel *s* in a
    # cluster of size *K* occupies the fractional band [s/K, (s+1)/K).
    # We use a fine grid approach to detect overlaps.

    _GRID_RESOLUTION: int = 1000  # sub-divide the total BW into this many bins

    # Map: grid_bin → list of (iot_idx, primary_uav_idx)
    grid_map: Dict[int, List[Tuple[int, int]]] = {}
    # Also record per-iot info: (primary_uav, sub_bw)
    iot_sub_bw: Dict[int, float] = {}

    for uav_idx, iot_list in cluster_assignments.items():
        K_j = len(iot_list)
        if K_j == 0:
            continue
        sub_bw = ofdma_subchannel_bandwidth(ch.W_UL_hz, K_j)
        for s_idx, iot_idx in enumerate(iot_list):
            iot_sub_bw[iot_idx] = sub_bw
            # Occupied grid bins
            lo_bin = int(s_idx / K_j * _GRID_RESOLUTION)
            hi_bin = int((s_idx + 1) / K_j * _GRID_RESOLUTION)
            for gb in range(lo_bin, hi_bin):
                grid_map.setdefault(gb, []).append((iot_idx, uav_idx))

    # For each IoT → UAV uplink, find cross-cluster interferers via grid
    for uav_idx, iot_list in cluster_assignments.items():
        K_j = len(iot_list)
        if K_j == 0:
            continue
        sub_bw = ofdma_subchannel_bandwidth(ch.W_UL_hz, K_j)

        for s_idx, iot_idx in enumerate(iot_list):
            # Grid bins for this device
            lo_bin = int(s_idx / K_j * _GRID_RESOLUTION)
            hi_bin = int((s_idx + 1) / K_j * _GRID_RESOLUTION)

            # Collect interferers (other-cluster IoTs on overlapping bins)
            interferer_set: set = set()
            for gb in range(lo_bin, hi_bin):
                for (other_iot, other_uav) in grid_map.get(gb, []):
                    if other_uav != uav_idx and other_iot != iot_idx:
                        interferer_set.add(other_iot)

            intrf_idx = list(interferer_set)
            intrf_pwr = [float(tx_powers[k]) for k in intrf_idx]

            sinr_val = compute_sinr(
                tx_idx=iot_idx,
                rx_idx=uav_idx,
                tx_power_W=float(tx_powers[iot_idx]),
                bandwidth_hz=sub_bw,
                gain_matrix=channel_state.gain_matrix,
                interferer_indices=intrf_idx,
                interferer_powers=intrf_pwr,
                N0_W_per_Hz=N0,
            )
            sinr_matrix[iot_idx, uav_idx] = sinr_val

    # --- UAV -> IoT downlink (each IoT gets W_DL / K_j) ---
    dl_grid_map: Dict[int, List[Tuple[int, int]]] = {}
    
    for uav_idx, iot_list in cluster_assignments.items():
        K_j = len(iot_list)
        if K_j == 0:
            continue
        for s_idx, iot_idx in enumerate(iot_list):
            lo_bin = int(s_idx / K_j * _GRID_RESOLUTION)
            hi_bin = int((s_idx + 1) / K_j * _GRID_RESOLUTION)
            for gb in range(lo_bin, hi_bin):
                dl_grid_map.setdefault(gb, []).append((uav_idx, iot_idx))

    for uav_idx, iot_list in cluster_assignments.items():
        K_j = len(iot_list)
        if K_j == 0:
            continue
        sub_bw_dl = ofdma_subchannel_bandwidth(ch.W_DL_hz, K_j)
        for s_idx, iot_idx in enumerate(iot_list):
            lo_bin = int(s_idx / K_j * _GRID_RESOLUTION)
            hi_bin = int((s_idx + 1) / K_j * _GRID_RESOLUTION)
            
            interferer_set = set()
            for gb in range(lo_bin, hi_bin):
                for (other_uav, other_iot) in dl_grid_map.get(gb, []):
                    if other_uav != uav_idx:
                        interferer_set.add(other_uav)
            
            other_uavs = list(interferer_set)
            intrf_pwr_dl = [float(tx_powers[u]) for u in other_uavs]
            
            sinr_dl = compute_sinr(
                tx_idx=uav_idx,
                rx_idx=iot_idx,
                tx_power_W=float(tx_powers[uav_idx]),
                bandwidth_hz=sub_bw_dl,
                gain_matrix=channel_state.gain_matrix,
                interferer_indices=other_uavs,
                interferer_powers=intrf_pwr_dl,
                N0_W_per_Hz=N0,
            )
            sinr_matrix[uav_idx, iot_idx] = sinr_dl

    # --- UAV ↔ UAV backhaul ---
    uav_indices = list(cluster_assignments.keys())
    for i_idx, uav_i in enumerate(uav_indices):
        for j_idx, uav_j in enumerate(uav_indices):
            if uav_i == uav_j:
                continue
            # Interferers: all other UAVs transmitting on backhaul
            other_uavs = [u for u in uav_indices if u != uav_i and u != uav_j]
            intrf_pwr_bh = [float(tx_powers[u]) for u in other_uavs]
            sinr_bh = compute_sinr(
                tx_idx=uav_i,
                rx_idx=uav_j,
                tx_power_W=float(tx_powers[uav_i]),
                bandwidth_hz=ch.W_backhaul_hz,
                gain_matrix=channel_state.gain_matrix,
                interferer_indices=other_uavs,
                interferer_powers=intrf_pwr_bh,
                N0_W_per_Hz=N0,
            )
            sinr_matrix[uav_i, uav_j] = sinr_bh

    return sinr_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running channel.py self-tests …")

    c = 299_792_458.0
    N0 = PhysicalConstants().N0_W_per_Hz

    # ----- 1. Free-space path loss at 100 m, 2.4 GHz -----
    pl = path_loss_free_space_dB(100.0, 2.4e9)
    assert 79.0 < pl < 82.0, f"FSPL at 100 m = {pl:.1f} dB — expected ~80 dB"
    print(f"  [OK] FSPL(100 m, 2.4 GHz) = {pl:.2f} dB")

    # ----- 2. Path loss at d = 0 -----
    assert path_loss_free_space_dB(0.0, 2.4e9) == 0.0
    print(f"  [OK] FSPL(0 m) = 0.0 dB")

    # ----- 3. Channel gain is positive and < 1 for d ≥ 1 m -----
    g = channel_gain(100.0, 2.4e9, 2.0, shadow_dB=0.0)
    assert 0.0 < g < 1.0, f"Gain = {g:.6e} — expected 0 < h < 1"
    print(f"  [OK] h(100 m, α=2) = {g:.6e}")

    # ----- 4. OFDMA bandwidth partitioning -----
    bw = ofdma_subchannel_bandwidth(20.0e6, 5)
    assert bw == 4.0e6, f"BW = {bw} — expected 4 MHz"
    print(f"  [OK] 20 MHz / 5 = {bw/1e6:.0f} MHz")

    # ----- 5. SINR degrades with interference -----
    gain_mat = np.array([
        [0.0, 1e-8, 1e-9],
        [1e-8, 0.0, 5e-9],
        [1e-9, 5e-9, 0.0],
    ])
    bw_test = 4.0e6

    sinr_no_int = compute_sinr(
        tx_idx=0, rx_idx=1, tx_power_W=0.1, bandwidth_hz=bw_test,
        gain_matrix=gain_mat, interferer_indices=[], interferer_powers=[],
        N0_W_per_Hz=N0,
    )

    sinr_with_int = compute_sinr(
        tx_idx=0, rx_idx=1, tx_power_W=0.1, bandwidth_hz=bw_test,
        gain_matrix=gain_mat, interferer_indices=[2], interferer_powers=[0.1],
        N0_W_per_Hz=N0,
    )

    assert sinr_with_int < sinr_no_int, (
        f"SINR should degrade: no_int={sinr_no_int:.2f}, with_int={sinr_with_int:.2f}"
    )
    print(
        f"  [OK] SINR degrades: no_int={sinr_no_int:.2f}, "
        f"with_int={sinr_with_int:.2f}"
    )

    # ----- 6. Shannon rate is positive and monotonic -----
    r1 = shannon_rate(bw_test, sinr_no_int)
    r2 = shannon_rate(bw_test, sinr_with_int)
    assert r1 > 0.0 and r2 > 0.0
    assert r1 > r2, f"Rate should be higher without interference: {r1:.0f} vs {r2:.0f}"
    print(f"  [OK] Shannon rate: no_int={r1/1e6:.2f} Mbps, with_int={r2/1e6:.2f} Mbps")

    # ----- 7. Interference telemetry -----
    telem = interference_impact_telemetry(
        tx_idx=0, rx_idx=1, tx_power_W=0.1, bandwidth_hz=bw_test,
        gain_matrix=gain_mat, interferer_indices=[2], interferer_powers=[0.1],
        N0_W_per_Hz=N0,
    )
    assert telem["degradation_ratio"] < 1.0, (
        f"Degradation ratio = {telem['degradation_ratio']:.4f} — expected < 1"
    )
    assert telem["interference_power_W"] > 0.0
    print(
        f"  [OK] Telemetry: degradation={telem['degradation_ratio']:.4f}, "
        f"I={telem['interference_power_W']:.2e} W"
    )

    # ----- 8. ChannelState integration -----
    rng = np.random.default_rng(42)
    positions = np.array([
        [0.0, 0.0, 0.0],       # IoT 0
        [100.0, 0.0, 0.0],     # IoT 1
        [50.0, 50.0, 100.0],   # UAV 0
        [500.0, 500.0, 250.0], # Cloud
    ])
    types = ["iot", "iot", "uav", "cloud"]
    cfg = SimulationConfig(num_iot=2, num_uav=1, num_cloud=1, seed=42)
    cs = ChannelState(positions, types, cfg, rng)

    assert cs.gain_matrix.shape == (4, 4)
    assert np.all(np.diag(cs.gain_matrix) == 0.0)  # no self-loops
    assert np.all(cs.gain_matrix >= 0.0)            # non-negative gains
    # Off-diagonal gains for non-trivial distances should be < 1
    for i in range(4):
        for j in range(4):
            if i != j:
                assert cs.gain_matrix[i, j] > 0.0, f"Zero gain at ({i},{j})"
    print(f"  [OK] ChannelState: shape={cs.gain_matrix.shape}, all gains > 0")

    print("[PASS] channel.py — All assertions passed.")
