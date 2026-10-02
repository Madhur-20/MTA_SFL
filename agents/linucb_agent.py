"""MTA-SFL Disjoint LinUCB Contextual Bandit — Algorithm 3, Eqs. 44–51.

Implements a per-primary-UAV LinUCB agent for multi-tier split-point
selection with:
- L₂-normalized 9D context extraction (Eq. 44)
- Disjoint per-arm covariance matrices with ridge regression (Eqs. 46–51)
- Upper Confidence Bound arm selection with dynamic masking
- Rank-1 Sherman-Morrison updates for stable online learning

References
----------
  Eq. 44:  Context vector extraction and normalization.
  Eq. 45:  LinUCB reward (see cooperative_reward.py).
  Eqs. 46–51:  Disjoint LinUCB model architecture and update rules.
  Lemma 2:  L₂ normalization guarantee ‖x‖₂ ≤ 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LinUCBConfig:
    """Configuration for Disjoint LinUCB agent.

    Attributes:
        context_dim: Dimensionality of the context vector (default 9).
        alpha_ucb: Exploration parameter α for UCB computation.
        use_sherman_morrison: If True, use rank-1 Sherman-Morrison for
            A⁻¹ updates; else recompute full inverse each step.
    """
    context_dim: int = 9
    alpha_ucb: float = 1.5
    use_sherman_morrison: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Context Extractor (Eq. 44)
# ═══════════════════════════════════════════════════════════════════════════

class ContextExtractor:
    """Extracts and L₂-normalizes the 9D context vector for LinUCB.

    Context features for pipeline i* → j → u → S:
        x = [E_res_i*, E_res_j, E_res_u, f_i*, f_j, f_u,
             R_{i*,j}^UL, R_{j,u}^UL, R_{u,S}^UL]

    Each feature is first normalized by its typical scale, then the
    full vector is L₂-normalized to satisfy Lemma 2: ‖x‖₂ ≤ 1.

    Args:
        E_init_iot: IoT initial energy for scale normalization (J).
        E_init_uav: UAV initial energy for scale normalization (J).
        f_max: Maximum CPU/GPU frequency for scale normalization (Hz).
        R_max: Maximum expected Shannon rate for scale normalization (bps).
    """

    def __init__(
        self,
        E_init_iot: float = 50_000.0,
        E_init_uav: float = 266_400.0,
        f_max: float = 4.0e9,
        R_max: float = 100.0e6,
    ) -> None:
        self.E_init_iot: float = E_init_iot
        self.E_init_uav: float = E_init_uav
        self.f_max: float = f_max
        self.R_max: float = R_max

    def extract_and_normalize(
        self,
        E_res_iot: float,
        E_res_primary: float,
        E_res_relay: float,
        f_iot: float,
        f_primary: float,
        f_relay: float,
        R_iot_primary: float,
        R_primary_relay: float,
        R_relay_cloud: float,
    ) -> np.ndarray:
        """Extract context and L₂-normalize to unit norm.

        Args:
            E_res_iot: IoT residual energy (J).
            E_res_primary: Primary UAV residual energy (J).
            E_res_relay: Relay UAV residual energy (J).
            f_iot: IoT CPU frequency (Hz).
            f_primary: Primary UAV GPU frequency (Hz).
            f_relay: Relay UAV GPU frequency (Hz).
            R_iot_primary: Shannon rate IoT→Primary (bps).
            R_primary_relay: Shannon rate Primary→Relay (bps).
            R_relay_cloud: Shannon rate Relay→Cloud (bps).

        Returns:
            L₂-normalized context vector of shape (9,) with ‖x‖₂ ≤ 1.0.
        """
        # Per-feature scale normalization
        raw = np.array([
            E_res_iot / (self.E_init_iot + 1e-8),
            E_res_primary / (self.E_init_uav + 1e-8),
            E_res_relay / (self.E_init_uav + 1e-8),
            f_iot / (self.f_max + 1e-8),
            f_primary / (self.f_max + 1e-8),
            f_relay / (self.f_max + 1e-8),
            R_iot_primary / (self.R_max + 1e-8),
            R_primary_relay / (self.R_max + 1e-8),
            R_relay_cloud / (self.R_max + 1e-8),
        ], dtype=np.float64)

        # L₂ normalization
        norm = np.linalg.norm(raw) + 1e-8
        normalized = raw / norm

        return normalized


# ═══════════════════════════════════════════════════════════════════════════
# 3. Disjoint LinUCB Agent (Eqs. 46–51)
# ═══════════════════════════════════════════════════════════════════════════

class DisjointLinUCBAgent:
    """Disjoint LinUCB contextual bandit agent.

    Maintains independent (A_p, b_p, A_inv_p) for each arm p ∈ P.

    Model: E[r | x, p] = x^T θ_p,  θ̂_p = A_p⁻¹ b_p

    UCB(p) = x^T θ̂_p + α √(x^T A_p⁻¹ x)

    Args:
        config: LinUCB configuration.
        arm_list: List of arm identifiers (e.g., split tuples).
    """

    def __init__(
        self,
        config: LinUCBConfig,
        arm_list: List[Tuple[int, int, int]],
    ) -> None:
        self.config: LinUCBConfig = config
        self.arm_list: List[Tuple[int, int, int]] = list(arm_list)
        self.num_arms: int = len(arm_list)
        self.d: int = config.context_dim

        # Per-arm model parameters
        self._A: List[np.ndarray] = []       # d × d covariance matrices
        self._b: List[np.ndarray] = []       # d bias vectors
        self._A_inv: List[np.ndarray] = []   # d × d inverse covariance

        # Per-arm statistics
        self._pulls: List[int] = []
        self._cumulative_reward: List[float] = []

        self._initialize_arms()

    def _initialize_arms(self) -> None:
        """Initialize all arm models to identity covariance and zero bias."""
        self._A.clear()
        self._b.clear()
        self._A_inv.clear()
        self._pulls.clear()
        self._cumulative_reward.clear()

        for _ in range(self.num_arms):
            self._A.append(np.eye(self.d, dtype=np.float64))
            self._b.append(np.zeros(self.d, dtype=np.float64))
            self._A_inv.append(np.eye(self.d, dtype=np.float64))
            self._pulls.append(0)
            self._cumulative_reward.append(0.0)

    # ----------------------------------------------------------------
    # UCB Computation
    # ----------------------------------------------------------------

    def compute_ucb(self, arm_idx: int, context: np.ndarray) -> float:
        """Compute the Upper Confidence Bound for one arm.

        UCB(p) = x^T θ̂_p + α √(x^T A_p⁻¹ x)

        Args:
            arm_idx: Index into the arm list.
            context: L₂-normalized context vector, shape (d,).

        Returns:
            UCB value for this arm.
        """
        A_inv = self._A_inv[arm_idx]
        b = self._b[arm_idx]

        # Ridge regression estimate: θ̂ = A⁻¹ b
        theta_hat = A_inv @ b

        # Predicted reward
        exploit = float(context @ theta_hat)

        # Confidence width
        explore_term = float(context @ A_inv @ context)
        explore = self.config.alpha_ucb * math.sqrt(max(explore_term, 0.0) + 1e-8)

        return exploit + explore

    # ----------------------------------------------------------------
    # Arm Selection with Masking
    # ----------------------------------------------------------------

    def select_arm(
        self,
        context: np.ndarray,
        feasibility_mask: np.ndarray,
    ) -> Tuple[int, float]:
        """Select the arm with highest UCB among feasible arms.

        Args:
            context: L₂-normalized context vector, shape (d,).
            feasibility_mask: Boolean mask, shape (num_arms,).

        Returns:
            Tuple of (arm_index, ucb_value).
            Returns (-1, -inf) if all arms are masked.
        """
        feasible_indices = np.where(feasibility_mask)[0]

        if len(feasible_indices) == 0:
            return -1, -float("inf")

        best_idx: int = -1
        best_ucb: float = -float("inf")

        for arm_idx in feasible_indices:
            ucb = self.compute_ucb(int(arm_idx), context)
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = int(arm_idx)

        return best_idx, best_ucb

    # ----------------------------------------------------------------
    # Model Update (Rank-1 Sherman-Morrison)
    # ----------------------------------------------------------------

    def update(
        self,
        arm_idx: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update the arm model with observed (context, reward) pair.

        A_p ← A_p + x x^T
        b_p ← b_p + r · x
        A_p⁻¹ updated via Sherman-Morrison rank-1 formula.

        Args:
            arm_idx: Index of the played arm.
            context: L₂-normalized context vector, shape (d,).
            reward: Observed reward.
        """
        x = context.reshape(-1, 1)  # (d, 1)

        # Update A
        self._A[arm_idx] = self._A[arm_idx] + x @ x.T

        # Update b
        self._b[arm_idx] = self._b[arm_idx] + reward * context

        # Update A⁻¹ via Sherman-Morrison
        if self.config.use_sherman_morrison:
            A_inv = self._A_inv[arm_idx]
            A_inv_x = A_inv @ x  # (d, 1)
            denominator = float((1.0 + x.T @ A_inv_x).item()) + 1e-8
            self._A_inv[arm_idx] = A_inv - (A_inv_x @ A_inv_x.T) / denominator
        else:
            # Full inverse (slower but numerically more stable)
            self._A_inv[arm_idx] = np.linalg.inv(
                self._A[arm_idx] + 1e-8 * np.eye(self.d)
            )

        # Update statistics
        self._pulls[arm_idx] += 1
        self._cumulative_reward[arm_idx] += reward

    # ----------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------

    def get_theta(self, arm_idx: int) -> np.ndarray:
        """Return current ridge regression estimate θ̂ for an arm.

        Args:
            arm_idx: Arm index.

        Returns:
            θ̂ vector of shape (d,).
        """
        return self._A_inv[arm_idx] @ self._b[arm_idx]

    def get_confidence_width(
        self, arm_idx: int, context: np.ndarray,
    ) -> float:
        """Return the confidence width α √(x^T A⁻¹ x) for an arm.

        Args:
            arm_idx: Arm index.
            context: Context vector, shape (d,).

        Returns:
            Confidence width value.
        """
        A_inv = self._A_inv[arm_idx]
        quadratic = float(context @ A_inv @ context)
        return self.config.alpha_ucb * math.sqrt(max(quadratic, 0.0) + 1e-8)

    def reset(self) -> None:
        """Reset all arm models to initial state."""
        self._initialize_arms()

    def get_arm_stats(self) -> Dict[int, Dict[str, float]]:
        """Per-arm statistics: pulls and mean reward estimate.

        Returns:
            Dict mapping arm_idx to stats dict.
        """
        stats: Dict[int, Dict[str, float]] = {}
        for i in range(self.num_arms):
            mean_reward = (self._cumulative_reward[i] / max(self._pulls[i], 1))
            stats[i] = {
                "pulls": float(self._pulls[i]),
                "cumulative_reward": self._cumulative_reward[i],
                "mean_reward": mean_reward,
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. LinUCB Split Allocator (High-level Wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class LinUCBSplitAllocator:
    """High-level wrapper managing the arm space and LinUCB agent.

    Generates the full space of valid split tuples (p1, p2, p3) for
    a model with L blocks and wraps the DisjointLinUCBAgent.

    Args:
        L: Total number of model blocks (e.g., 12).
        config: LinUCB configuration.
        context_extractor: Context extraction and normalization.
    """

    def __init__(
        self,
        L: int,
        config: LinUCBConfig,
        context_extractor: ContextExtractor,
    ) -> None:
        self.L: int = L
        self.config: LinUCBConfig = config
        self.context_extractor: ContextExtractor = context_extractor

        # Generate arm space: all (p1, p2, p3) with 1 ≤ p1 ≤ p2 ≤ p3 ≤ L
        self.arm_list: List[Tuple[int, int, int]] = self._generate_arms()

        # Create agent
        self.agent: DisjointLinUCBAgent = DisjointLinUCBAgent(
            config, self.arm_list,
        )

    def _generate_arms(self) -> List[Tuple[int, int, int]]:
        """Generate all valid split tuples."""
        arms: List[Tuple[int, int, int]] = []
        for p1 in range(1, self.L + 1):
            for p2 in range(p1, self.L + 1):
                for p3 in range(p2, self.L + 1):
                    arms.append((p1, p2, p3))
        return arms

    def select_split(
        self,
        context: np.ndarray,
        feasibility_mask: np.ndarray,
    ) -> Tuple[Tuple[int, int, int], int, float]:
        """Select the best split configuration from feasible arms.

        Args:
            context: L₂-normalized context vector.
            feasibility_mask: Boolean mask over arm space.

        Returns:
            Tuple of (split_config, arm_index, ucb_value).
            Returns ((0,0,0), -1, -inf) if no feasible arm exists.
        """
        arm_idx, ucb_val = self.agent.select_arm(context, feasibility_mask)
        if arm_idx == -1:
            return (0, 0, 0), -1, -float("inf")
        return self.arm_list[arm_idx], arm_idx, ucb_val

    def update_with_reward(
        self,
        arm_idx: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update the played arm with observed reward.

        Args:
            arm_idx: Index of the played arm.
            context: Context vector used during selection.
            reward: Observed reward.
        """
        self.agent.update(arm_idx, context, reward)

    def get_num_arms(self) -> int:
        """Total number of arms in the split space."""
        return len(self.arm_list)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running linucb_agent.py self-tests …")

    # ---- Setup ----
    L = 12
    config = LinUCBConfig(context_dim=9, alpha_ucb=1.5)
    extractor = ContextExtractor()

    allocator = LinUCBSplitAllocator(L=L, config=config, context_extractor=extractor)

    # ---- Test 1: Arm space ----
    num_arms = allocator.get_num_arms()
    # For L=12: C(12+2, 3) = C(14,3) = 364 combinations
    expected = sum(1 for p1 in range(1, 13)
                   for p2 in range(p1, 13)
                   for p3 in range(p2, 13))
    assert num_arms == expected, f"Arms: {num_arms}, expected {expected}"
    print(f"  [OK] Arm space: {num_arms} valid split tuples for L={L}")

    assert allocator.arm_list[0] == (1, 1, 1)
    assert allocator.arm_list[-1] == (12, 12, 12)
    print(f"  [OK] First arm: {allocator.arm_list[0]}, Last: {allocator.arm_list[-1]}")

    # ---- Test 2: Context extraction & normalization ----
    ctx = extractor.extract_and_normalize(
        E_res_iot=30000.0,
        E_res_primary=200000.0,
        E_res_relay=150000.0,
        f_iot=500e6,
        f_primary=2e9,
        f_relay=2e9,
        R_iot_primary=10e6,
        R_primary_relay=20e6,
        R_relay_cloud=20e6,
    )

    assert ctx.shape == (9,), f"Context shape = {ctx.shape}, expected (9,)"
    ctx_norm = float(np.linalg.norm(ctx))
    assert ctx_norm <= 1.0 + 1e-6, f"‖x‖₂ = {ctx_norm}, should be ≤ 1.0"
    print(f"  [OK] Context shape = {ctx.shape}, ‖x‖₂ = {ctx_norm:.6f}")

    # ---- Test 3: UCB computation ----
    ucb_val = allocator.agent.compute_ucb(0, ctx)
    assert math.isfinite(ucb_val), f"UCB should be finite, got {ucb_val}"
    print(f"  [OK] UCB(arm 0) = {ucb_val:.6f}")

    # ---- Test 4: Arm selection with mask ----
    full_mask = np.ones(num_arms, dtype=bool)
    split, arm_idx, ucb = allocator.select_split(ctx, full_mask)
    assert arm_idx >= 0, f"Expected valid arm, got {arm_idx}"
    assert split in allocator.arm_list
    print(f"  [OK] Selected split {split} (arm {arm_idx}, UCB={ucb:.6f})")

    # Partial mask
    partial_mask = np.zeros(num_arms, dtype=bool)
    partial_mask[10] = True
    partial_mask[50] = True
    split_p, idx_p, ucb_p = allocator.select_split(ctx, partial_mask)
    assert idx_p in [10, 50], f"Expected arm 10 or 50, got {idx_p}"
    print(f"  [OK] Partial mask: selected arm {idx_p}")

    # All-masked
    empty_mask = np.zeros(num_arms, dtype=bool)
    split_e, idx_e, ucb_e = allocator.select_split(ctx, empty_mask)
    assert idx_e == -1, f"Expected -1 for empty mask, got {idx_e}"
    print("  [OK] Empty mask returns arm_idx = -1")

    # ---- Test 5: Update and confidence shrinking ----
    # Record initial confidence width
    cw_before = allocator.agent.get_confidence_width(0, ctx)

    for _ in range(20):
        allocator.update_with_reward(0, ctx, reward=-0.3)

    cw_after = allocator.agent.get_confidence_width(0, ctx)
    assert cw_after < cw_before, \
        f"Confidence should shrink: {cw_before:.6f} → {cw_after:.6f}"
    print(f"  [OK] Confidence shrinks: {cw_before:.6f} → {cw_after:.6f}")

    # ---- Test 6: Sherman-Morrison vs direct inverse ----
    agent_sm = DisjointLinUCBAgent(
        LinUCBConfig(context_dim=9, alpha_ucb=1.0, use_sherman_morrison=True),
        allocator.arm_list,
    )
    agent_direct = DisjointLinUCBAgent(
        LinUCBConfig(context_dim=9, alpha_ucb=1.0, use_sherman_morrison=False),
        allocator.arm_list,
    )

    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.normal(size=9).astype(np.float64)
        x = x / (np.linalg.norm(x) + 1e-8)
        r = rng.normal()
        agent_sm.update(0, x, r)
        agent_direct.update(0, x, r)

    # Compare inverses
    diff = np.max(np.abs(agent_sm._A_inv[0] - agent_direct._A_inv[0]))
    assert diff < 1e-8, f"Sherman-Morrison vs direct inverse diff = {diff}"
    print(f"  [OK] Sherman-Morrison matches direct inverse (max diff = {diff:.2e})")

    # ---- Test 7: Theta estimate ----
    theta = allocator.agent.get_theta(0)
    assert theta.shape == (9,), f"θ shape = {theta.shape}"
    assert np.all(np.isfinite(theta)), "θ should be finite"
    print(f"  [OK] θ̂(arm 0) = {theta[:3]}... (shape {theta.shape})")

    # ---- Test 8: Arm stats ----
    stats = allocator.agent.get_arm_stats()
    assert 0 in stats
    assert stats[0]["pulls"] == 20.0, f"Arm 0 pulls = {stats[0]['pulls']}"
    print(f"  [OK] Arm stats: arm 0 pulls = {int(stats[0]['pulls'])}")

    # ---- Test 9: Reset ----
    allocator.agent.reset()
    cw_reset = allocator.agent.get_confidence_width(0, ctx)
    assert abs(cw_reset - cw_before) < 1e-6, "Reset should restore initial confidence"
    print("  [OK] Reset restores initial state")

    # ---- Test 10: 100-round simulated bandit ----
    allocator2 = LinUCBSplitAllocator(L=6, config=config, context_extractor=extractor)
    num_arms_6 = allocator2.get_num_arms()
    print(f"\n  Simulated bandit with L=6 ({num_arms_6} arms):")

    # Define a true best arm for this context
    true_rewards = rng.normal(loc=-0.5, scale=0.1, size=num_arms_6)
    best_true_arm = int(np.argmax(true_rewards))
    true_rewards[best_true_arm] = 0.0  # best arm has reward ~0

    mask_all = np.ones(num_arms_6, dtype=bool)
    selected_best_count = 0

    for t in range(100):
        ctx_t = extractor.extract_and_normalize(
            30000 + rng.normal() * 1000,
            200000 + rng.normal() * 5000,
            150000 + rng.normal() * 5000,
            500e6, 2e9, 2e9,
            10e6 + rng.normal() * 1e6,
            20e6 + rng.normal() * 2e6,
            20e6 + rng.normal() * 2e6,
        )
        split_t, arm_t, ucb_t = allocator2.select_split(ctx_t, mask_all)
        reward_t = true_rewards[arm_t] + rng.normal() * 0.05
        allocator2.update_with_reward(arm_t, ctx_t, reward_t)

        if arm_t == best_true_arm:
            selected_best_count += 1

    print(f"  Best-arm selection rate: {selected_best_count}/100")
    print(f"  True best arm: {best_true_arm} → split {allocator2.arm_list[best_true_arm]}")

    # Check some stats
    final_stats = allocator2.agent.get_arm_stats()
    total_pulls = sum(s["pulls"] for s in final_stats.values())
    assert total_pulls == 100, f"Total pulls = {total_pulls}, expected 100"
    print(f"  [OK] Total pulls = {int(total_pulls)}")

    print("\n[PASS] linucb_agent.py — All assertions passed.")
