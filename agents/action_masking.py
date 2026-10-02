"""MTA-SFL Dynamic Physical Feasibility Masking — Sec. VI.C, Eq. 38.

Implements strict dynamic action masking for both the DDQN relay selector
and the LinUCB workload allocator, ensuring that agents never execute
physically infeasible or constraint-violating actions.

DDQN Mask:
    A_feasible(t) = N_j(t)  ∩  U_cloud(t)  ∩  U_safe(t)   (Eq. 38)

LinUCB Mask:
    Filters the split-tuple arm space P by estimated pipeline latency
    (Constraint 33b) and per-node energy (Constraint 33c).

References
----------
  Eq. 38:  Feasible relay action space construction.
  Sec. VI.C:  Topological bridge masking and constraint enforcement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from mta_sfl.config.system_config import SimulationConfig
from mta_sfl.physics.energy import EnergyLedger
from mta_sfl.physics.mobility import Entity3D, compute_route_lifetime
from mta_sfl.physics.topology import TopologyManager


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pipeline Latency Estimator
# ═══════════════════════════════════════════════════════════════════════════

def estimate_pipeline_latency(
    p: Tuple[int, int, int],
    L: int,
    f_iot: float,
    f_uav: float,
    f_cloud: float,
    flops_per_block: float,
    rates: Dict[str, float],
    cluster_size: int = 1,
) -> float:
    """Estimate total pipeline latency for a split configuration."""
    p1, p2, p3 = p

    # Computation latency per tier
    T_comp_tier1 = (p1 * flops_per_block) / (f_iot + 1e-30)
    T_comp_tier2 = ((p2 - p1) * flops_per_block * cluster_size) / (f_uav + 1e-30)
    T_comp_tier3 = ((p3 - p2) * flops_per_block * cluster_size) / (f_uav + 1e-30)
    T_comp_tier4 = ((L - p3) * flops_per_block * cluster_size) / (f_cloud + 1e-30)

    # Communication latency (assume intermediate activation size proportional
    # to a base data_size that is transmitted between tiers; use 1 Mbit as
    # a representative smashed-data payload between each tier boundary)
    _BASE_DATA_BITS: float = 1.0e6  # 1 Mbit activation transfer

    R_iot_primary = rates.get("R_iot_primary", 1e6)
    R_primary_relay = rates.get("R_primary_relay", 1e6)
    R_relay_cloud = rates.get("R_relay_cloud", 1e6)

    T_comm_1 = _BASE_DATA_BITS / (R_iot_primary + 1e-30) if p1 < L else 0.0
    T_comm_2 = _BASE_DATA_BITS / (R_primary_relay + 1e-30) if p2 < L else 0.0
    T_comm_3 = _BASE_DATA_BITS / (R_relay_cloud + 1e-30) if p3 < L else 0.0

    T_total = (T_comp_tier1 + T_comm_1 + T_comp_tier2 + T_comm_2
               + T_comp_tier3 + T_comm_3 + T_comp_tier4)
    return T_total


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pipeline Energy Estimator
# ═══════════════════════════════════════════════════════════════════════════

def estimate_pipeline_energy(
    p: Tuple[int, int, int],
    L: int,
    cfg: SimulationConfig,
    rates: Dict[str, float],
    flops_per_block: float,
    cluster_size: int = 1,
) -> Dict[str, float]:
    """Estimate per-node energy consumption for a split configuration."""
    p1, p2, p3 = p
    cmp = cfg.compute
    pwr = cfg.power

    # ---- Computation energy: E_comp = kappa * f^2 * FLOPs ----
    E_comp_iot = cmp.kappa * (cmp.f_iot_hz ** 2) * (p1 * flops_per_block)
    E_comp_primary = cmp.kappa * (cmp.f_uav_hz ** 2) * ((p2 - p1) * flops_per_block * cluster_size)
    E_comp_relay = cmp.kappa * (cmp.f_uav_hz ** 2) * ((p3 - p2) * flops_per_block * cluster_size)

    # ---- Communication energy ----
    _BASE_DATA_BITS: float = 1.0e6

    R_iot_primary = rates.get("R_iot_primary", 1e6)
    R_primary_relay = rates.get("R_primary_relay", 1e6)
    R_relay_cloud = rates.get("R_relay_cloud", 1e6)

    # IoT transmit time
    t_tx_iot = _BASE_DATA_BITS / (R_iot_primary + 1e-30) if p1 < L else 0.0
    E_comm_iot = pwr.P_tx_iot * t_tx_iot

    # Primary UAV: receives from IoT, transmits to relay
    t_rx_primary = t_tx_iot
    t_tx_primary = _BASE_DATA_BITS / (R_primary_relay + 1e-30) if p2 < L else 0.0
    E_comm_primary = pwr.P_rx_uav * t_rx_primary + pwr.P_tx_uav * t_tx_primary

    # Relay UAV: receives from primary, transmits to cloud
    t_rx_relay = t_tx_primary
    t_tx_relay = _BASE_DATA_BITS / (R_relay_cloud + 1e-30) if p3 < L else 0.0
    E_comm_relay = pwr.P_rx_uav * t_rx_relay + pwr.P_tx_uav * t_tx_relay

    # ---- Estimate total pipeline time for circuit/propulsion ----
    T_est = estimate_pipeline_latency(
        p, L, cmp.f_iot_hz, cmp.f_uav_hz, cmp.f_cloud_hz,
        flops_per_block, rates, cluster_size
    )

    # ---- Circuit + propulsion for UAVs (over estimated pipeline time) ----
    E_circuit_primary = pwr.P_circuit_uav * T_est
    E_prop_primary = pwr.P_flight_hover * T_est
    E_circuit_relay = pwr.P_circuit_uav * T_est
    E_prop_relay = pwr.P_flight_hover * T_est

    # ---- IoT circuit (active for its computation + comm time) ----
    T_iot_active = (p1 * flops_per_block) / (cmp.f_iot_hz + 1e-30) + t_tx_iot
    E_circuit_iot = pwr.P_circuit_iot_active * T_iot_active

    # ---- Totals ----
    E_total_iot = E_comp_iot + E_comm_iot + E_circuit_iot
    E_total_primary = E_comp_primary + E_comm_primary + E_circuit_primary + E_prop_primary
    E_total_relay = E_comp_relay + E_comm_relay + E_circuit_relay + E_prop_relay

    return {
        "iot": E_total_iot,
        "primary": E_total_primary,
        "relay": E_total_relay,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. DDQN Action Mask  (Eq. 38)
# ═══════════════════════════════════════════════════════════════════════════

class DDQNActionMask:
    """Constructs the boolean feasibility mask for DDQN relay selection.

    Enforces Eq. 38:
        A_feasible(t) = N_j(t) ∩ U_cloud(t) ∩ U_safe(t),  u ≠ j

    Args:
        num_uavs: Total number of UAVs in the network.
        uav_indices: Global node indices corresponding to UAV nodes.
        cfg: Simulation configuration.
    """

    def __init__(
        self,
        num_uavs: int,
        uav_indices: List[int],
        cfg: SimulationConfig,
    ) -> None:
        self.num_uavs: int = num_uavs
        self.uav_indices: List[int] = list(uav_indices)
        self.cfg: SimulationConfig = cfg

    def compute_mask(
        self,
        primary_uav_idx: int,
        sinr_matrix: np.ndarray,
        energy_ledger: EnergyLedger,
        entities: List[Entity3D],
        cloud_indices: Set[int],
        topology: TopologyManager,
        T_max: float,
    ) -> Tuple[np.ndarray, bool]:
        """Compute the DDQN feasibility mask for relay selection.

        Args:
            primary_uav_idx: Global index of the querying primary UAV.
            sinr_matrix: N×N linear SINR matrix.
            energy_ledger: Current energy state.
            entities: All simulation entities (indexed by global ID).
            cloud_indices: Set of cloud gateway global indices.
            topology: Current topology manager (already updated).
            T_max: Maximum allowable pipeline latency (s).

        Returns:
            Tuple of:
            - Boolean mask of shape ``(num_uavs,)`` where True = feasible.
            - ``is_routing_failure``: True if no feasible action exists.
        """
        mask = np.zeros(self.num_uavs, dtype=bool)

        # Compute cloud-reachable set
        cloud_reachable: Set[int] = topology.cloud_reachable_set(cloud_indices)

        # Get topological neighbors that are also cloud-reachable: N_j ∩ U_cloud
        relay_candidates: List[int] = topology.get_relay_candidates(
            primary_uav_idx, cloud_reachable
        )

        gamma_safe: float = self.cfg.battery.gamma_safe
        R_comm: float = self.cfg.mobility.R_comm_m

        for local_idx, uav_global_idx in enumerate(self.uav_indices):
            # Cannot select self
            if uav_global_idx == primary_uav_idx:
                continue

            # Must be in N_j ∩ U_cloud
            if uav_global_idx not in relay_candidates:
                continue

            # U_safe: battery check — E_res >= gamma_safe * E_init
            e_fraction = energy_ledger.get_fraction(uav_global_idx)
            if e_fraction < gamma_safe:
                continue

            # U_safe: link stability — L_life >= T_max
            entity_j = entities[primary_uav_idx]
            entity_u = entities[uav_global_idx]
            L_life = compute_route_lifetime(entity_j, entity_u, R_comm)
            if L_life < T_max:
                continue

            # Passed all checks — mark as feasible
            mask[local_idx] = True

        is_routing_failure: bool = not np.any(mask)
        return mask, is_routing_failure


# ═══════════════════════════════════════════════════════════════════════════
# 4. LinUCB Action Mask  (Sec. VI.C)
# ═══════════════════════════════════════════════════════════════════════════

class LinUCBActionMask:
    """Constructs the boolean feasibility mask for LinUCB split selection.

    Enforces:
    - Constraint 33b: T̂_total(p) ≤ T_max
    - Constraint 33c: Ê_total,n(p) < E_res,n  for all active nodes
    - Topological bridge: if relay is critical, force p2 = p3

    Args:
        L: Total number of model blocks (e.g. 12).
        cfg: Simulation configuration.
    """

    def __init__(self, L: int, cfg: SimulationConfig) -> None:
        self.L: int = L
        self.cfg: SimulationConfig = cfg

    def generate_arm_space(self) -> List[Tuple[int, int, int]]:
        """Generate all valid split tuples (p1, p2, p3).

        Returns:
            Sorted list of tuples satisfying 1 ≤ p1 ≤ p2 ≤ p3 ≤ L.
        """
        arms: List[Tuple[int, int, int]] = []
        for p1 in range(1, self.L + 1):
            for p2 in range(p1, self.L + 1):
                for p3 in range(p2, self.L + 1):
                    arms.append((p1, p2, p3))
        return arms

    def compute_mask(
        self,
        arms: List[Tuple[int, int, int]],
        iot_idx: int,
        primary_uav_idx: int,
        relay_uav_idx: int,
        energy_ledger: EnergyLedger,
        rates: Dict[str, float],
        T_max: float,
        flops_per_block: float,
        cluster_size: int = 1,
        reserved_energy_map: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute the LinUCB feasibility mask for split selection."""
        num_arms: int = len(arms)
        mask = np.ones(num_arms, dtype=bool)
        attempted_violations: int = 0
        
        reserved_relay_energy = reserved_energy_map.get(relay_uav_idx, 0.0) if reserved_energy_map else 0.0

        # Check if relay is energy-critical → force bridge mode
        relay_critical: bool = energy_ledger.is_critical(relay_uav_idx)

        E_res_iot: float = energy_ledger.get_residual(iot_idx)
        E_res_primary: float = energy_ledger.get_residual(primary_uav_idx)
        E_res_relay: float = energy_ledger.get_residual(relay_uav_idx) - reserved_relay_energy

        cmp = self.cfg.compute

        for arm_idx, p in enumerate(arms):
            p1, p2, p3 = p

            # ---- Topological bridge masking ----
            if relay_critical and p2 != p3:
                mask[arm_idx] = False
                attempted_violations += 1
                continue

            # ---- Latency constraint (33b) ----
            T_est = estimate_pipeline_latency(
                p, self.L,
                cmp.f_iot_hz, cmp.f_uav_hz, cmp.f_cloud_hz,
                flops_per_block, rates, cluster_size
            )
            if T_est > T_max:
                mask[arm_idx] = False
                attempted_violations += 1
                continue

            # ---- Energy constraint (33c) ----
            E_est = estimate_pipeline_energy(
                p, self.L, self.cfg, rates, flops_per_block, cluster_size
            )

            if E_est["iot"] >= E_res_iot:
                mask[arm_idx] = False
                attempted_violations += 1
                continue
            if E_est["primary"] >= E_res_primary:
                mask[arm_idx] = False
                attempted_violations += 1
                continue
            if E_est["relay"] >= E_res_relay:
                mask[arm_idx] = False
                attempted_violations += 1
                continue

        return mask, attempted_violations


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running action_masking.py self-tests …")

    from mta_sfl.config.system_config import SimulationConfig
    from mta_sfl.physics.energy import EnergyLedger
    from mta_sfl.physics.mobility import UAVNode, CloudGateway, IoTDevice
    from mta_sfl.physics.topology import TopologyManager

    cfg = SimulationConfig(num_iot=2, num_uav=4, num_cloud=1, seed=42)

    # ---- Setup entities ----
    # Indices: 0,1 = IoT; 2,3,4,5 = UAV; 6 = Cloud
    entities: List[Entity3D] = [
        IoTDevice(0, 100.0, 100.0),
        IoTDevice(1, 200.0, 200.0),
        UAVNode(2, 150.0, 150.0, 100.0, vx=1.0, vy=0.0),  # primary
        UAVNode(3, 160.0, 150.0, 100.0, vx=0.5, vy=0.0),  # relay candidate (close)
        UAVNode(4, 900.0, 900.0, 100.0, vx=0.0, vy=0.0),  # far away
        UAVNode(5, 170.0, 150.0, 100.0, vx=0.0, vy=0.0),  # relay candidate (close)
        CloudGateway(6, 500.0, 500.0, 200.0),
    ]
    uav_indices = [2, 3, 4, 5]

    # ---- Setup SINR matrix ----
    # Make UAV 2 (primary) connected to UAV 3, 5 but not 4
    # All UAVs connected to cloud
    N = 7
    sinr = np.zeros((N, N))
    gamma_th = cfg.channel.sinr_threshold_linear  # ~2.0

    # Primary (2) ↔ relay candidates (3, 5): SINR > threshold
    sinr[2, 3] = 10.0; sinr[3, 2] = 10.0
    sinr[2, 5] = 8.0;  sinr[5, 2] = 8.0
    sinr[2, 4] = 0.5   # below threshold → no link

    # UAVs ↔ cloud (6)
    sinr[3, 6] = 5.0; sinr[6, 3] = 5.0
    sinr[5, 6] = 5.0; sinr[6, 5] = 5.0
    sinr[4, 6] = 5.0; sinr[6, 4] = 5.0  # even far UAV can reach cloud
    sinr[2, 6] = 5.0; sinr[6, 2] = 5.0

    # UAV ↔ UAV backhaul links
    sinr[3, 5] = 6.0; sinr[5, 3] = 6.0

    # Build topology
    topology = TopologyManager(cfg, num_nodes=N)
    topology.update(sinr)

    # ---- Setup energy ledger ----
    ledger = EnergyLedger(num_iot=2, num_uav=4, num_cloud=1, cfg=cfg)

    # Deplete UAV 4 to below gamma_safe
    ledger.deduct(4, ledger.E_initial[4] * 0.90)  # 10% remaining < 15% threshold

    cloud_indices: Set[int] = {6}

    # ---- Test 1: DDQN mask ----
    ddqn_mask = DDQNActionMask(
        num_uavs=4, uav_indices=uav_indices, cfg=cfg,
    )

    mask, routing_failure = ddqn_mask.compute_mask(
        primary_uav_idx=2,
        sinr_matrix=sinr,
        energy_ledger=ledger,
        entities=entities,
        cloud_indices=cloud_indices,
        topology=topology,
        T_max=5.0,
    )

    print(f"  DDQN mask: {mask}")
    print(f"  Routing failure: {routing_failure}")

    # UAV 2 (self) → should be masked out
    assert not mask[0], "Self (UAV 2) should be masked"
    # UAV 4 (far, below gamma_safe) → should be masked out
    assert not mask[2], "Depleted UAV 4 should be masked"
    print("  [OK] DDQN mask filters self and depleted nodes")

    # ---- Test 2: LinUCB mask ----
    linucb_mask_obj = LinUCBActionMask(L=12, cfg=cfg)
    arms = linucb_mask_obj.generate_arm_space()
    print(f"\n  LinUCB total arms (L=12): {len(arms)}")
    assert len(arms) > 0, "Should have arms"
    # Verify arm ordering
    assert arms[0] == (1, 1, 1), f"First arm = {arms[0]}, expected (1,1,1)"
    assert arms[-1] == (12, 12, 12), f"Last arm = {arms[-1]}, expected (12,12,12)"

    rates = {
        "R_iot_primary": 2e6,   # 2 Mbps  (tight link)
        "R_primary_relay": 5e6,  # 5 Mbps
        "R_relay_cloud": 5e6,   # 5 Mbps
    }
    # Use a large flops_per_block so IoT-heavy splits violate latency
    flops_per_block = cfg.compute.flops_per_sample * 5000  # 5000 samples/block

    linucb_mask, violations = linucb_mask_obj.compute_mask(
        arms=arms,
        iot_idx=0,
        primary_uav_idx=2,
        relay_uav_idx=3,
        energy_ledger=ledger,
        rates=rates,
        T_max=2.0,  # tight deadline
        flops_per_block=flops_per_block,
    )

    feasible_count = int(np.sum(linucb_mask))
    print(f"  LinUCB feasible: {feasible_count}/{len(arms)}, violations masked: {violations}")
    assert feasible_count < len(arms), "Some arms should be masked"
    assert violations > 0, "Should have masked some violations"
    print("  [OK] LinUCB mask filters infeasible splits")

    # ---- Test 3: Bridge masking ----
    # Deplete relay (UAV 3) to critical level
    ledger_bridge = EnergyLedger(num_iot=2, num_uav=4, num_cloud=1, cfg=cfg)
    ledger_bridge.deduct(3, ledger_bridge.E_initial[3] * 0.90)  # 10% remaining

    mask_bridge, viols_bridge = linucb_mask_obj.compute_mask(
        arms=arms,
        iot_idx=0,
        primary_uav_idx=2,
        relay_uav_idx=3,
        energy_ledger=ledger_bridge,
        rates=rates,
        T_max=100.0,  # very relaxed latency
        flops_per_block=flops_per_block,
    )

    # Check that all feasible arms have p2 == p3 (bridge mode)
    for arm_idx, is_feasible in enumerate(mask_bridge):
        if is_feasible:
            p1, p2, p3 = arms[arm_idx]
            assert p2 == p3, f"Bridge masking failed: arm {arms[arm_idx]} should have p2==p3"

    print(f"  Bridge masking: {int(np.sum(mask_bridge))} feasible (all p2==p3)")
    print("  [OK] Topological bridge masking works correctly")

    # ---- Test 4: Latency estimator ----
    T_est = estimate_pipeline_latency(
        (3, 6, 9), 12,
        cfg.compute.f_iot_hz, cfg.compute.f_uav_hz, cfg.compute.f_cloud_hz,
        flops_per_block, rates,
    )
    assert T_est > 0.0, f"Latency should be positive: {T_est}"
    assert math.isfinite(T_est), f"Latency should be finite: {T_est}"
    print(f"\n  Latency estimate (3,6,9): {T_est:.6f} s")
    print("  [OK] Latency estimator returns finite positive value")

    # ---- Test 5: Energy estimator ----
    E_est = estimate_pipeline_energy(
        (3, 6, 9), 12, cfg, rates, flops_per_block,
    )
    for role, e_val in E_est.items():
        assert e_val >= 0.0, f"Energy for {role} should be non-negative: {e_val}"
        assert math.isfinite(e_val), f"Energy for {role} should be finite: {e_val}"
    print(f"  Energy estimates: IoT={E_est['iot']:.4f}J, "
          f"Primary={E_est['primary']:.4f}J, Relay={E_est['relay']:.4f}J")
    print("  [OK] Energy estimator returns valid values")

    print("\n[PASS] action_masking.py — All assertions passed.")
