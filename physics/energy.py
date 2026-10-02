"""MTA-SFL Strict Energy Ledger & Global Synchronisation Barrier.

Implements the paper's energy accounting equations:

* **Eq. 25** — Global sync barrier:  T_global = max_k T_{total,k}
* **Eq. 26** — Propulsion energy:    E_prop  = P_flight · T_global
* **Eq. 27** — UAV circuit energy:   E_cir   = P_cir   · T_global
* **Eq. 28** — IoT circuit energy:   E_cir,i = P_cir·T_active + P_sleep·(T_global − T_active)
* **Eqs. 29–31** — RF energy:        E_rf    = P_tx·Δt_tx + P_rx·Δt_rx
* **Eq. 32** — Residual update:      E_res(t+1) = max(E_res(t) − E_total(t), 0)

Key design decision: **no asynchronous hovering**.  All aerial nodes consume
propulsion and circuit energy for the *entire* duration of ``T_global``,
enforcing the synchronous barrier mandated by the paper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mta_sfl.config.system_config import (
    BatteryConfig,
    ComputeConfig,
    PowerConfig,
    SimulationConfig,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Global Synchronisation Barrier  (Eq. 25)
# ──────────────────────────────────────────────────────────────────────────────
class GlobalSyncBarrier:
    """Compute the global synchronisation time ``T_global``.

    All participating entities must wait for the slowest task to complete
    before the round ends.  Every aerial node therefore burns propulsion and
    circuit energy for this full duration.
    """

    def compute_T_global(self, task_durations: Dict[int, float]) -> float:
        """T_global(t) = max_{k ∈ J}  T_{total,k}(t)     (Eq. 25).

        Args:
            task_durations: Mapping node_id → total task duration (s).

        Returns:
            Maximum duration among all tasks (s).

        Raises:
            ValueError: If *task_durations* is empty.
        """
        if not task_durations:
            raise ValueError("task_durations must not be empty")
        return max(task_durations.values())

    def compute_T_global_from_list(self, durations: List[float]) -> float:
        """Convenience overload accepting a flat list.

        Args:
            durations: List of task durations (s).

        Returns:
            Maximum duration (s).

        Raises:
            ValueError: If *durations* is empty.
        """
        if not durations:
            raise ValueError("durations list must not be empty")
        return max(durations)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Component Energy Functions  (Eqs. 26–31)
# ──────────────────────────────────────────────────────────────────────────────
def compute_propulsion_energy(P_flight: float, T_global: float) -> float:
    """UAV propulsion energy for one global round  (Eq. 26).

    E_prop = P_flight · T_global

    Args:
        P_flight: Rotor mechanical power, hover or cruise (W).
        T_global: Global barrier duration (s).

    Returns:
        Propulsion energy (J).
    """
    return P_flight * T_global


def compute_circuit_energy_uav(P_circuit: float, T_global: float) -> float:
    """UAV active circuit energy for one global round  (Eq. 27).

    E_cir = P_circuit · T_global

    The UAV is powered on for the *entire* barrier duration.

    Args:
        P_circuit: UAV hardware circuit power (W).
        T_global: Global barrier duration (s).

    Returns:
        Circuit energy (J).
    """
    return P_circuit * T_global


def compute_circuit_energy_iot(
    P_circuit_active: float,
    P_sleep: float,
    T_active: float,
    T_global: float,
) -> float:
    """IoT device circuit energy with active / sleep split  (Eq. 28).

    E_cir,i = P_cir · T_active  +  P_sleep · (T_global − T_active)

    If T_active > T_global (shouldn't happen, but guard), it is clamped.

    Args:
        P_circuit_active: Active-state circuit power (W).
        P_sleep: Idle / sleep power (W).
        T_active: Duration of active computation / communication (s).
        T_global: Global barrier duration (s).

    Returns:
        Circuit energy (J).
    """
    T_active = min(T_active, T_global)
    T_idle = T_global - T_active
    return P_circuit_active * T_active + P_sleep * T_idle


def compute_rf_energy(
    P_tx: float,
    t_tx: float,
    P_rx: float,
    t_rx: float,
) -> float:
    """Directional RF transmission + reception energy  (Eqs. 29–31).

    E_rf = P_tx · Δt_tx  +  P_rx · Δt_rx

    Covers both uplink and downlink legs of a link; call once per
    directional transfer with the appropriate power and duration.

    Args:
        P_tx: Transmit power (W).
        t_tx: Transmit duration (s).
        P_rx: Receive chain power (W).
        t_rx: Receive duration (s).

    Returns:
        Total RF energy (J).
    """
    return P_tx * t_tx + P_rx * t_rx


def compute_computation_energy(
    kappa: float,
    f_hz: float,
    flops: float,
) -> float:
    """On-device computation energy.

    E_comp = κ · f² · FLOPs

    Args:
        kappa: Effective switched capacitance coefficient (F).
        f_hz: CPU / GPU clock frequency (Hz).
        flops: Total floating-point operations executed.

    Returns:
        Computation energy (J).
    """
    return kappa * (f_hz ** 2) * flops


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Energy Ledger  (Eq. 32 + accounting)
# ──────────────────────────────────────────────────────────────────────────────
class EnergyLedger:
    """Per-node battery state manager for the entire simulation.

    Node indexing convention:
        [0, num_iot)                             → IoT devices
        [num_iot, num_iot + num_uav)             → UAV nodes
        [num_iot + num_uav, num_iot + num_uav + num_cloud) → Cloud gateways

    Attributes:
        E_initial:  Initial energy per node (J).
        E_residual: Current residual energy per node (J).
        E_min:      Minimum safe energy per node (J).
    """

    def __init__(
        self,
        num_iot: int,
        num_uav: int,
        num_cloud: int,
        cfg: SimulationConfig,
    ) -> None:
        self.num_iot = num_iot
        self.num_uav = num_uav
        self.num_cloud = num_cloud
        self.num_total = num_iot + num_uav + num_cloud

        bat = cfg.battery

        self.E_initial = np.zeros(self.num_total, dtype=np.float64)
        self.E_min = np.zeros(self.num_total, dtype=np.float64)

        # IoT
        self.E_initial[:num_iot] = bat.E_init_iot_J
        self.E_min[:num_iot] = bat.E_min_iot_J

        # UAV
        uav_lo = num_iot
        uav_hi = num_iot + num_uav
        self.E_initial[uav_lo:uav_hi] = bat.E_init_uav_J
        self.E_min[uav_lo:uav_hi] = bat.E_min_uav_J

        # Cloud
        cloud_lo = uav_hi
        cloud_hi = self.num_total
        self.E_initial[cloud_lo:cloud_hi] = bat.E_init_cloud_J
        self.E_min[cloud_lo:cloud_hi] = bat.E_min_cloud_J

        self.E_residual = self.E_initial.copy()
        self._gamma_safe = bat.gamma_safe

    # --------------------------------------------------------- single deduct
    def deduct(self, node_idx: int, energy_J: float) -> None:
        """Deduct energy from a single node  (Eq. 32).

        E_res[n] = max(E_res[n] − energy, 0.0)

        Emits a WARNING log if the deduction crosses the E_min threshold.

        Args:
            node_idx: Node index.
            energy_J: Energy to deduct (J).
        """
        new_val = self.E_residual[node_idx] - energy_J
        if new_val < self.E_min[node_idx] and self.E_residual[node_idx] >= self.E_min[node_idx]:
            logger.warning(
                "Node %d crossed E_min: %.1f J → %.1f J  (E_min=%.1f J)",
                node_idx, self.E_residual[node_idx], new_val, self.E_min[node_idx],
            )
        self.E_residual[node_idx] = max(new_val, 0.0)

    # --------------------------------------------------------- batch deduct
    def deduct_round(self, round_energies: Dict[int, float]) -> None:
        """Batch deduction for all nodes in a round.

        Args:
            round_energies: Mapping node_id → energy consumed (J).
        """
        for node_idx, energy in round_energies.items():
            self.deduct(node_idx, energy)

    # --------------------------------------------------------- full round
    def apply_global_round(
        self,
        T_global: float,
        task_durations: Dict[int, float],
        rf_durations: Dict[int, Tuple[float, float, float, float]],
        comp_flops: Dict[int, float],
        node_types: Dict[int, str],
        cfg: SimulationConfig,
    ) -> Dict[int, float]:
        """Apply one full global round of energy accounting.

        For every node, computes and deducts the sum of all applicable
        energy components (propulsion, circuit, RF, compute).

        **Critical invariant:** All UAVs consume propulsion and circuit
        energy for the *entire* ``T_global`` duration, not just their own
        task duration.

        Args:
            T_global: Global synchronisation barrier duration (s).
            task_durations: node_id → active task duration (s).
            rf_durations: node_id → (P_tx, t_tx, P_rx, t_rx).
            comp_flops: node_id → total FLOPs computed.
            node_types: node_id → ``'iot'`` / ``'uav'`` / ``'cloud'``.
            cfg: Simulation configuration.

        Returns:
            Mapping node_id → total energy consumed in this round (J).
        """
        pwr = cfg.power
        cmp = cfg.compute
        round_energies: Dict[int, float] = {}

        for node_id, ntype in node_types.items():
            T_active = task_durations.get(node_id, 0.0)
            rf = rf_durations.get(node_id, (0.0, 0.0, 0.0, 0.0))
            flops = comp_flops.get(node_id, 0.0)

            e_rf = compute_rf_energy(*rf)

            if ntype == "uav":
                # Propulsion for FULL T_global  (Eq. 26)
                e_prop = compute_propulsion_energy(pwr.P_flight_hover, T_global)
                # Circuit for FULL T_global  (Eq. 27)
                e_cir = compute_circuit_energy_uav(pwr.P_circuit_uav, T_global)
                e_comp = compute_computation_energy(cmp.kappa, cmp.f_uav_hz, flops)
                total = e_prop + e_cir + e_rf + e_comp

            elif ntype == "iot":
                # Active + sleep split  (Eq. 28)
                e_cir = compute_circuit_energy_iot(
                    pwr.P_circuit_iot_active, pwr.P_sleep_iot, T_active, T_global,
                )
                e_comp = compute_computation_energy(cmp.kappa, cmp.f_iot_hz, flops)
                total = e_cir + e_rf + e_comp

            elif ntype == "cloud":
                # Cloud is grid-powered; track for accounting
                e_cir = pwr.P_circuit_cloud * T_global
                e_comp = compute_computation_energy(cmp.kappa, cmp.f_cloud_hz, flops)
                total = e_cir + e_rf + e_comp

            else:
                raise ValueError(f"Unknown node type: {ntype!r}")

            round_energies[node_id] = total
            self.deduct(node_id, total)

        return round_energies

    # --------------------------------------------------------- queries
    def get_residual(self, node_idx: int) -> float:
        """Current residual energy (J)."""
        return float(self.E_residual[node_idx])

    def get_all_residuals(self) -> np.ndarray:
        """Copy of residual energies for all nodes."""
        return self.E_residual.copy()

    def get_fraction(self, node_idx: int) -> float:
        """Residual fraction E_res / E_init."""
        return float(self.E_residual[node_idx] / self.E_initial[node_idx])

    def is_alive(self, node_idx: int) -> bool:
        """True if residual energy is above the safe minimum."""
        return bool(self.E_residual[node_idx] > self.E_min[node_idx])

    def is_critical(self, node_idx: int) -> bool:
        """True if residual fraction is below ``gamma_safe``."""
        return self.get_fraction(node_idx) < self._gamma_safe

    def get_depletion_rate(
        self,
        node_idx: int,
        energy_consumed: float,
        dt: float,
    ) -> float:
        """Instantaneous depletion rate (W).

        Args:
            node_idx: Node index (unused except for future extensions).
            energy_consumed: Energy consumed in the interval (J).
            dt: Interval duration (s).

        Returns:
            Power draw (W).  Returns 0.0 if dt ≤ 0.
        """
        if dt <= 0.0:
            return 0.0
        return energy_consumed / dt

    def alive_mask(self) -> np.ndarray:
        """Boolean mask: True for nodes with E_res > E_min."""
        return self.E_residual > self.E_min

    def summary(self) -> Dict[str, Any]:
        """Aggregate battery health statistics.

        Returns:
            Dictionary with keys ``avg_fraction``, ``min_fraction``,
            ``num_alive``, ``num_critical``, ``total_nodes``.
        """
        fractions = self.E_residual / self.E_initial
        alive = self.alive_mask()
        return {
            "avg_fraction": float(np.mean(fractions)),
            "min_fraction": float(np.min(fractions)),
            "num_alive": int(np.sum(alive)),
            "num_critical": int(np.sum(fractions < self._gamma_safe)),
            "total_nodes": self.num_total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running energy.py self-tests …")

    cfg = SimulationConfig(num_iot=4, num_uav=3, num_cloud=1, seed=7)

    # ----- 1. GlobalSyncBarrier -----
    barrier = GlobalSyncBarrier()

    durations = {0: 5.0, 1: 10.0, 2: 7.5}
    T_g = barrier.compute_T_global(durations)
    assert T_g == 10.0, f"T_global = {T_g}, expected 10.0"
    print(f"  [OK] T_global (dict) = {T_g}")

    T_g2 = barrier.compute_T_global_from_list([3.0, 8.0, 2.0])
    assert T_g2 == 8.0
    print(f"  [OK] T_global (list) = {T_g2}")

    try:
        barrier.compute_T_global({})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] Empty dict raises ValueError")

    # ----- 2. Propulsion energy  (Eq. 26) -----
    e_prop = compute_propulsion_energy(80.0, 10.0)
    assert e_prop == 800.0, f"E_prop = {e_prop}, expected 800.0"
    print(f"  [OK] E_prop = {e_prop} J  (80 W × 10 s)")

    # ----- 3. UAV circuit energy  (Eq. 27) -----
    e_cir_uav = compute_circuit_energy_uav(5.0, 10.0)
    assert e_cir_uav == 50.0
    print(f"  [OK] E_cir_uav = {e_cir_uav} J  (5 W × 10 s)")

    # ----- 4. IoT circuit energy  (Eq. 28) -----
    e_cir_iot = compute_circuit_energy_iot(2.0, 0.05, 4.0, 10.0)
    expected_iot = 2.0 * 4.0 + 0.05 * (10.0 - 4.0)  # 8.0 + 0.3 = 8.3
    assert abs(e_cir_iot - expected_iot) < 1e-12, (
        f"E_cir_iot = {e_cir_iot}, expected {expected_iot}"
    )
    print(f"  [OK] E_cir_iot = {e_cir_iot} J  (active + sleep)")

    # T_active clamped to T_global
    e_clamped = compute_circuit_energy_iot(2.0, 0.05, 15.0, 10.0)
    assert e_clamped == 2.0 * 10.0  # all active, no sleep
    print(f"  [OK] E_cir_iot (clamped) = {e_clamped} J")

    # ----- 5. RF energy  (Eqs. 29–31) -----
    e_rf = compute_rf_energy(0.5, 3.0, 0.3, 4.0)
    expected_rf = 0.5 * 3.0 + 0.3 * 4.0  # 1.5 + 1.2 = 2.7
    assert abs(e_rf - expected_rf) < 1e-12
    print(f"  [OK] E_rf = {e_rf} J")

    # ----- 6. Computation energy -----
    e_comp = compute_computation_energy(1e-27, 2e9, 1e6)
    expected_comp = 1e-27 * (2e9) ** 2 * 1e6  # 4.0e-12 × 1e6 = 4.0e-6
    assert abs(e_comp - expected_comp) < 1e-15
    print(f"  [OK] E_comp = {e_comp:.6e} J  (κ·f²·FLOPs)")

    # ----- 7. EnergyLedger basic operations  (Eq. 32) -----
    ledger = EnergyLedger(
        num_iot=4, num_uav=3, num_cloud=1, cfg=cfg,
    )

    # Check initial state
    assert ledger.get_residual(0) == cfg.battery.E_init_iot_J
    assert ledger.get_residual(4) == cfg.battery.E_init_uav_J  # first UAV
    assert ledger.get_residual(7) == cfg.battery.E_init_cloud_J  # Cloud
    assert ledger.get_fraction(0) == 1.0
    assert ledger.is_alive(0)
    assert not ledger.is_critical(0)
    print("  [OK] EnergyLedger initial state")

    # Deduction
    ledger.deduct(0, 40_000.0)
    assert ledger.get_residual(0) == 10_000.0
    assert ledger.is_alive(0)  # 10 k > 5 k
    ledger.deduct(0, 6_000.0)
    assert ledger.get_residual(0) == 4_000.0
    assert not ledger.is_alive(0)  # 4 k ≤ 5 k
    print("  [OK] EnergyLedger deduction + is_alive")

    # Never below zero
    ledger.deduct(0, 999_999.0)
    assert ledger.get_residual(0) == 0.0
    print("  [OK] E_residual never goes below 0.0")

    # is_critical
    frac = ledger.get_fraction(0)
    assert frac == 0.0
    assert ledger.is_critical(0)
    print("  [OK] is_critical detects depleted node")

    # ----- 8. Global round: T_global dictates ALL UAV propulsion -----
    ledger2 = EnergyLedger(num_iot=2, num_uav=2, num_cloud=1, cfg=cfg)

    node_types = {0: "iot", 1: "iot", 2: "uav", 3: "uav", 4: "cloud"}
    task_dur = {0: 3.0, 1: 5.0, 2: 8.0, 3: 2.0, 4: 1.0}
    T_gl = barrier.compute_T_global(task_dur)
    assert T_gl == 8.0

    rf_dur: Dict[int, Tuple[float, float, float, float]] = {
        0: (0.1, 1.0, 0.1, 1.0),
        1: (0.1, 1.5, 0.1, 1.5),
        2: (0.5, 2.0, 0.3, 2.0),
        3: (0.5, 1.0, 0.3, 1.0),
        4: (2.0, 0.5, 0.3, 0.5),
    }
    comp = {0: 1e6, 1: 1e6, 2: 2e6, 3: 2e6, 4: 5e6}

    energies = ledger2.apply_global_round(
        T_gl, task_dur, rf_dur, comp, node_types, cfg,
    )

    # Both UAVs must have propulsion = P_flight_hover × T_global
    pwr = cfg.power
    expected_prop = pwr.P_flight_hover * T_gl  # 80 × 8 = 640 J
    expected_cir_uav = pwr.P_circuit_uav * T_gl  # 5 × 8 = 40 J

    # UAV 2's energy must include 640 + 40 + RF + comp
    e2_rf = compute_rf_energy(*rf_dur[2])
    e2_comp = compute_computation_energy(cfg.compute.kappa, cfg.compute.f_uav_hz, comp[2])
    expected_e2 = expected_prop + expected_cir_uav + e2_rf + e2_comp
    assert abs(energies[2] - expected_e2) < 1e-6, (
        f"UAV 2: {energies[2]:.6f} ≠ {expected_e2:.6f}"
    )

    # UAV 3 (only active 2 s) still burns propulsion for 8 s
    e3_rf = compute_rf_energy(*rf_dur[3])
    e3_comp = compute_computation_energy(cfg.compute.kappa, cfg.compute.f_uav_hz, comp[3])
    expected_e3 = expected_prop + expected_cir_uav + e3_rf + e3_comp
    assert abs(energies[3] - expected_e3) < 1e-6, (
        f"UAV 3: {energies[3]:.6f} ≠ {expected_e3:.6f}"
    )
    print(
        f"  [OK] Global round: both UAVs consume prop={expected_prop} J "
        f"for T_global={T_gl} s"
    )

    # IoT 0 (active 3 s, idle 5 s):
    e0_cir = compute_circuit_energy_iot(
        pwr.P_circuit_iot_active, pwr.P_sleep_iot, 3.0, T_gl,
    )
    e0_rf = compute_rf_energy(*rf_dur[0])
    e0_comp = compute_computation_energy(cfg.compute.kappa, cfg.compute.f_iot_hz, comp[0])
    expected_e0 = e0_cir + e0_rf + e0_comp
    assert abs(energies[0] - expected_e0) < 1e-6
    print(f"  [OK] IoT 0: active 3 s + sleep 5 s = {e0_cir:.4f} J circuit")

    # ----- 9. Alive mask & summary -----
    mask = ledger2.alive_mask()
    assert mask.shape == (5,)
    assert all(mask)  # all still alive after one round
    s = ledger2.summary()
    assert s["num_alive"] == 5
    assert s["total_nodes"] == 5
    assert 0.0 < s["avg_fraction"] <= 1.0
    print(f"  [OK] Summary: {s}")

    print("[PASS] energy.py — All assertions passed.")
