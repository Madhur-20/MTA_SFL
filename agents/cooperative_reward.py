"""MTA-SFL Cooperative Reward Decomposition — Eqs. 33, 39, 45.

Implements the unified reward structure aligning local agent rewards
with the global potential function, ensuring cooperative Nash equilibrium
convergence (Theorem 1).

Global Cost (Eq. 33):
    C(a) = λ₁·T̃_total + λ₂·Ẽ_ratio − λ₃·CC_u

DDQN Extrinsic Reward (Eq. 39):
    r_ext = λ₃·CC_u − λ₂·Ẽ_ratio − λ₁·T̃_total − Ψ

LinUCB Local Reward (Eq. 45):
    r_local = −(λ₁·T̃_total + λ₂·Ẽ_ratio)

References
----------
  Eq. 33:  Global cost potential function.
  Eq. 39:  DDQN extrinsic reward with terminal penalty Ψ.
  Eq. 45:  LinUCB local reward aligned with global potential Φ = −C.
  Eq. 63:  Extended global cost with clustering coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reward Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RewardConfig:
    """Multi-objective reward weights and penalty parameters.

    Attributes:
        lambda_1: Weight for normalized latency.
        lambda_2: Weight for energy consumption ratio.
        lambda_3: Weight for topology (clustering coefficient).
        psi_penalty: Terminal failure penalty M.
        T_max: Maximum allowable pipeline latency (s).
    """
    lambda_1: float = 0.4
    lambda_2: float = 0.3
    lambda_3: float = 0.3
    psi_penalty: float = 50.0
    T_max: float = 10.0

    def __post_init__(self) -> None:
        """Validate that lambda weights sum to 1.0 (within tolerance)."""
        total = self.lambda_1 + self.lambda_2 + self.lambda_3
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Lambda weights must sum to 1.0, got "
                f"{self.lambda_1} + {self.lambda_2} + {self.lambda_3} = {total}"
            )
        if self.T_max <= 0.0:
            raise ValueError(f"T_max must be positive, got {self.T_max}")
        if self.psi_penalty < 0.0:
            raise ValueError(f"psi_penalty must be non-negative, got {self.psi_penalty}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cooperative Reward Computer
# ═══════════════════════════════════════════════════════════════════════════

class CooperativeRewardComputer:
    """Unified reward decomposition for cooperative multi-agent MTA-SFL.

    Computes global cost (Eq. 33), DDQN extrinsic reward (Eq. 39),
    LinUCB local reward (Eq. 45), and global potential Φ = −C.

    Args:
        config: Reward configuration with lambda weights and penalties.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config: RewardConfig = config

    # ----------------------------------------------------------------
    # Normalization helpers
    # ----------------------------------------------------------------

    def normalize_latency(self, T_total: float) -> float:
        """Normalize latency to [0, 1]: T̃ = T_total / T_max, clipped.

        Args:
            T_total: Raw pipeline latency (s).

        Returns:
            Normalized latency ∈ [0, 1].
        """
        return float(np.clip(T_total / self.config.T_max, 0.0, 1.0))

    def compute_energy_ratio(
        self,
        per_node_energy: Dict[int, float],
        per_node_residual: Dict[int, float],
    ) -> float:
        """Compute average energy consumption ratio across active nodes.

        Ẽ_ratio = (1/|A_active|) · Σ_{n ∈ A_active} (E_total,n / E_res,n)

        Adds 1e-8 to residual for numerical stability.

        Args:
            per_node_energy: node_id → energy consumed this round (J).
            per_node_residual: node_id → residual energy (J).

        Returns:
            Average energy ratio. Returns 0.0 if no active nodes.
        """
        if not per_node_energy:
            return 0.0

        total_ratio: float = 0.0
        count: int = 0

        for node_id, e_consumed in per_node_energy.items():
            e_res = per_node_residual.get(node_id, 1e-8)
            ratio = e_consumed / (e_res + 1e-8)
            total_ratio += ratio
            count += 1

        if count == 0:
            return 0.0

        return total_ratio / count

    # ----------------------------------------------------------------
    # Global Cost (Eq. 33 & 63)
    # ----------------------------------------------------------------

    def compute_global_cost(
        self,
        T_total: float,
        per_node_energy: Dict[int, float],
        per_node_residual: Dict[int, float],
        cc_relay: float,
    ) -> float:
        """Global cost potential function (Eq. 33).

        C(a) = λ₁·T̃_total + λ₂·Ẽ_ratio − λ₃·CC_u

        Args:
            T_total: Raw pipeline latency (s).
            per_node_energy: Per-node energy consumed (J).
            per_node_residual: Per-node residual energy (J).
            cc_relay: Clustering coefficient of selected relay ∈ [0, 1].

        Returns:
            Global cost value (lower is better).
        """
        cfg = self.config
        T_norm = self.normalize_latency(T_total)
        E_ratio = self.compute_energy_ratio(per_node_energy, per_node_residual)

        cost = (cfg.lambda_1 * T_norm
                + cfg.lambda_2 * E_ratio
                - cfg.lambda_3 * cc_relay)
        return cost

    # ----------------------------------------------------------------
    # DDQN Extrinsic Reward (Eq. 39)
    # ----------------------------------------------------------------

    def compute_ddqn_reward(
        self,
        T_total: float,
        per_node_energy: Dict[int, float],
        per_node_residual: Dict[int, float],
        cc_relay: float,
        constraint_violated: bool,
        route_failed: bool,
    ) -> float:
        """DDQN extrinsic reward with terminal penalty (Eq. 39).

        r_ext = λ₃·CC_u − λ₂·Ẽ_ratio − λ₁·T̃_total − Ψ

        where Ψ = M if any constraint is violated or route fails, else 0.

        Args:
            T_total: Raw pipeline latency (s).
            per_node_energy: Per-node energy consumed (J).
            per_node_residual: Per-node residual energy (J).
            cc_relay: Clustering coefficient of selected relay.
            constraint_violated: True if any hard constraint was violated.
            route_failed: True if routing failed (no feasible relay).

        Returns:
            DDQN extrinsic reward value.
        """
        cfg = self.config
        T_norm = self.normalize_latency(T_total)
        E_ratio = self.compute_energy_ratio(per_node_energy, per_node_residual)

        # Terminal penalty
        psi: float = cfg.psi_penalty if (constraint_violated or route_failed) else 0.0

        reward = (cfg.lambda_3 * cc_relay
                  - cfg.lambda_2 * E_ratio
                  - cfg.lambda_1 * T_norm
                  - psi)
        return reward

    # ----------------------------------------------------------------
    # LinUCB Local Reward (Eq. 45)
    # ----------------------------------------------------------------

    def compute_linucb_reward(
        self,
        T_total: float,
        per_node_energy: Dict[int, float],
        per_node_residual: Dict[int, float],
    ) -> float:
        """LinUCB local reward aligned with global potential (Eq. 45).

        r_local = −(λ₁·T̃_total + λ₂·Ẽ_ratio)

        This ensures monotonic alignment with Φ(a) = −C(a).

        Args:
            T_total: Raw pipeline latency (s).
            per_node_energy: Per-node energy consumed (J).
            per_node_residual: Per-node residual energy (J).

        Returns:
            LinUCB local reward (negative cost, higher = better).
        """
        cfg = self.config
        T_norm = self.normalize_latency(T_total)
        E_ratio = self.compute_energy_ratio(per_node_energy, per_node_residual)

        return -(cfg.lambda_1 * T_norm + cfg.lambda_2 * E_ratio)

    # ----------------------------------------------------------------
    # Global Potential Φ = −C
    # ----------------------------------------------------------------

    def compute_global_potential(
        self,
        T_total: float,
        per_node_energy: Dict[int, float],
        per_node_residual: Dict[int, float],
        cc_relay: float,
    ) -> float:
        """Global potential function Φ(a) = −C(a).

        Higher potential → better action. Used to verify reward alignment.

        Args:
            T_total: Raw pipeline latency (s).
            per_node_energy: Per-node energy consumed (J).
            per_node_residual: Per-node residual energy (J).
            cc_relay: Clustering coefficient of selected relay.

        Returns:
            Potential value (higher is better).
        """
        return -self.compute_global_cost(
            T_total, per_node_energy, per_node_residual, cc_relay
        )

    # ----------------------------------------------------------------
    # Alignment Diagnostics
    # ----------------------------------------------------------------

    def verify_reward_alignment(
        self,
        ddqn_reward: float,
        linucb_reward: float,
        global_cost: float,
    ) -> Dict[str, float]:
        """Diagnostic comparison of local rewards vs global potential.

        Args:
            ddqn_reward: DDQN extrinsic reward value.
            linucb_reward: LinUCB local reward value.
            global_cost: Global cost C(a).

        Returns:
            Dict with alignment diagnostics:
            - ``global_potential``: −C(a)
            - ``ddqn_reward``: DDQN reward
            - ``linucb_reward``: LinUCB reward
            - ``ddqn_vs_potential``: r_ddqn − Φ (deviation)
            - ``linucb_vs_potential``: r_linucb − (−C + λ₃·CC) (deviation)
        """
        potential = -global_cost

        return {
            "global_potential": potential,
            "ddqn_reward": ddqn_reward,
            "linucb_reward": linucb_reward,
            "ddqn_vs_potential": ddqn_reward - potential,
            "linucb_vs_potential": linucb_reward - potential,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running cooperative_reward.py self-tests …")

    # ---- Test 1: Lambda normalization ----
    rc = RewardConfig(lambda_1=0.4, lambda_2=0.3, lambda_3=0.3)
    print(f"  [OK] RewardConfig created: λ = ({rc.lambda_1}, {rc.lambda_2}, {rc.lambda_3})")

    # Invalid lambdas should raise
    try:
        RewardConfig(lambda_1=0.5, lambda_2=0.5, lambda_3=0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [OK] Invalid lambda sum correctly raises ValueError")

    crc = CooperativeRewardComputer(rc)

    # ---- Test 2: Latency normalization ----
    assert crc.normalize_latency(0.0) == 0.0
    assert crc.normalize_latency(5.0) == 0.5  # T_max = 10
    assert crc.normalize_latency(10.0) == 1.0
    assert crc.normalize_latency(15.0) == 1.0  # clipped
    assert crc.normalize_latency(-1.0) == 0.0  # clipped
    print("  [OK] Latency normalization with clipping")

    # ---- Test 3: Energy ratio ----
    per_energy = {0: 100.0, 1: 200.0, 2: 50.0}
    per_residual = {0: 10000.0, 1: 5000.0, 2: 20000.0}

    E_ratio = crc.compute_energy_ratio(per_energy, per_residual)
    expected = (100.0 / 10000.0 + 200.0 / 5000.0 + 50.0 / 20000.0) / 3.0
    assert abs(E_ratio - expected) < 1e-10, f"E_ratio = {E_ratio}, expected {expected}"
    print(f"  [OK] Energy ratio = {E_ratio:.6f}")

    # Empty dict
    assert crc.compute_energy_ratio({}, {}) == 0.0
    print("  [OK] Empty energy ratio = 0.0")

    # ---- Test 4: Numerical stability with near-zero residual ----
    per_residual_tiny = {0: 0.0, 1: 1e-20}
    per_energy_tiny = {0: 1.0, 1: 1.0}
    E_ratio_tiny = crc.compute_energy_ratio(per_energy_tiny, per_residual_tiny)
    assert np.isfinite(E_ratio_tiny), f"E_ratio should be finite, got {E_ratio_tiny}"
    print(f"  [OK] Numerical stability: E_ratio = {E_ratio_tiny:.4f} (near-zero residual)")

    # ---- Test 5: Global cost ----
    cost = crc.compute_global_cost(
        T_total=5.0,
        per_node_energy=per_energy,
        per_node_residual=per_residual,
        cc_relay=0.8,
    )
    T_norm = 0.5  # 5 / 10
    expected_cost = 0.4 * T_norm + 0.3 * E_ratio - 0.3 * 0.8
    assert abs(cost - expected_cost) < 1e-10, f"Cost = {cost}, expected {expected_cost}"
    print(f"  [OK] Global cost = {cost:.6f}")

    # ---- Test 6: DDQN reward without penalty ----
    r_ddqn = crc.compute_ddqn_reward(
        T_total=5.0,
        per_node_energy=per_energy,
        per_node_residual=per_residual,
        cc_relay=0.8,
        constraint_violated=False,
        route_failed=False,
    )
    expected_ddqn = 0.3 * 0.8 - 0.3 * E_ratio - 0.4 * T_norm
    assert abs(r_ddqn - expected_ddqn) < 1e-10, f"DDQN reward = {r_ddqn}"
    print(f"  [OK] DDQN reward (no penalty) = {r_ddqn:.6f}")

    # ---- Test 7: DDQN reward WITH penalty ----
    r_ddqn_fail = crc.compute_ddqn_reward(
        T_total=5.0,
        per_node_energy=per_energy,
        per_node_residual=per_residual,
        cc_relay=0.8,
        constraint_violated=True,
        route_failed=False,
    )
    assert r_ddqn_fail == r_ddqn - 50.0, f"Penalty not applied: {r_ddqn_fail}"
    print(f"  [OK] DDQN reward with Ψ = {r_ddqn_fail:.6f} (penalty applied)")

    r_ddqn_route_fail = crc.compute_ddqn_reward(
        T_total=5.0,
        per_node_energy=per_energy,
        per_node_residual=per_residual,
        cc_relay=0.8,
        constraint_violated=False,
        route_failed=True,
    )
    assert r_ddqn_route_fail == r_ddqn - 50.0
    print("  [OK] Route failure triggers same penalty")

    # ---- Test 8: LinUCB reward ----
    r_linucb = crc.compute_linucb_reward(
        T_total=5.0,
        per_node_energy=per_energy,
        per_node_residual=per_residual,
    )
    expected_linucb = -(0.4 * T_norm + 0.3 * E_ratio)
    assert abs(r_linucb - expected_linucb) < 1e-10
    print(f"  [OK] LinUCB reward = {r_linucb:.6f}")

    # ---- Test 9: Alignment — lower cost ↔ higher reward ----
    # Action A: low latency, low energy
    cost_A = crc.compute_global_cost(2.0, {0: 50.0}, {0: 10000.0}, 0.9)
    r_ddqn_A = crc.compute_ddqn_reward(2.0, {0: 50.0}, {0: 10000.0}, 0.9, False, False)
    r_linucb_A = crc.compute_linucb_reward(2.0, {0: 50.0}, {0: 10000.0})

    # Action B: high latency, high energy
    cost_B = crc.compute_global_cost(8.0, {0: 500.0}, {0: 10000.0}, 0.2)
    r_ddqn_B = crc.compute_ddqn_reward(8.0, {0: 500.0}, {0: 10000.0}, 0.2, False, False)
    r_linucb_B = crc.compute_linucb_reward(8.0, {0: 500.0}, {0: 10000.0})

    assert cost_A < cost_B, f"Action A should have lower cost: {cost_A} vs {cost_B}"
    assert r_ddqn_A > r_ddqn_B, f"Action A should have higher DDQN reward"
    assert r_linucb_A > r_linucb_B, f"Action A should have higher LinUCB reward"
    print("  [OK] Reward alignment: lower cost ↔ higher rewards for both agents")

    # ---- Test 10: Global potential ----
    phi = crc.compute_global_potential(5.0, per_energy, per_residual, 0.8)
    assert abs(phi + cost) < 1e-10, "Φ should equal −C"
    print(f"  [OK] Global potential Φ = {phi:.6f} = −C")

    # ---- Test 11: Alignment diagnostics ----
    diag = crc.verify_reward_alignment(r_ddqn, r_linucb, cost)
    assert "global_potential" in diag
    assert "ddqn_reward" in diag
    assert "linucb_reward" in diag
    print(f"  [OK] Alignment diagnostics: {diag}")

    # ---- Test 12: Edge case T = 0 ----
    r_zero = crc.compute_ddqn_reward(0.0, {}, {}, 1.0, False, False)
    assert np.isfinite(r_zero)
    print(f"  [OK] Edge case T=0: reward = {r_zero:.6f}")

    # ---- Test 13: Edge case T > T_max ----
    r_over = crc.compute_ddqn_reward(100.0, {0: 1.0}, {0: 100.0}, 0.5, False, False)
    assert np.isfinite(r_over)
    T_clipped = crc.normalize_latency(100.0)
    assert T_clipped == 1.0
    print(f"  [OK] Edge case T>T_max: latency clipped to 1.0, reward = {r_over:.6f}")

    print("\n[PASS] cooperative_reward.py — All assertions passed.")
