"""MTA-SFL Chaos Engineering and Robustness Simulator.

Simulates sudden hardware failures, deep fading, and network blockages
to validate resilience and recovery (Phase 5). Addresses Reviewer 4's
request for stress tests under sudden UAV failures.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class ChaosInjector:
    """Injects stochastic failures into the physical environment."""

    def __init__(self, seed: int = 42) -> None:
        """
        Args:
            seed: Random seed for reproducible chaos.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def inject_hardware_failures(
        self,
        e_residual: np.ndarray,
        active_uav_indices: List[int],
        p_fail: float
    ) -> List[int]:
        """Randomly forces residual battery to 0.0 for active UAVs.
        
        This forces the DDQN/LinUCB action masks to collapse and tests
        the topological bridge/repair mechanics.

        Args:
            e_residual: Array of residual energy per node (modified in-place).
            active_uav_indices: List of currently active UAV indices.
            p_fail: Probability of failure per UAV.

        Returns:
            List of UAV indices that failed.
        """
        failed_uavs: List[int] = []
        for uav_idx in active_uav_indices:
            if self.rng.random() < p_fail:
                e_residual[uav_idx] = 0.0
                failed_uavs.append(uav_idx)
        return failed_uavs

    def inject_deep_fading(
        self,
        channel_gains: np.ndarray,
        active_links: List[Tuple[int, int]],
        p_block: float,
        fading_factor: float = 1e-6
    ) -> List[Tuple[int, int]]:
        """Degrades channel gain heavily on random active links.

        Simulates obstacles causing SINR << gamma_th, forcing dropped batches
        and topology recalculations.

        Args:
            channel_gains: 2D array of channel gains (modified in-place).
            active_links: List of (tx, rx) index pairs currently communicating.
            p_block: Probability of blockage per link.
            fading_factor: Multiplier to degrade the channel.

        Returns:
            List of (tx, rx) links that suffered deep fading.
        """
        blocked_links: List[Tuple[int, int]] = []
        for tx, rx in active_links:
            if self.rng.random() < p_block:
                channel_gains[tx, rx] *= fading_factor
                # Assuming symmetry for simplicity if matrix is symmetric
                if tx != rx:
                    channel_gains[rx, tx] *= fading_factor
                blocked_links.append((tx, rx))
        return blocked_links

    def log_recovery_metrics(
        self,
        round_idx: int,
        failed_nodes: List[int],
        blocked_links: List[Tuple[int, int]],
        latency_before: float,
        latency_after: float,
        accuracy_drop: float
    ) -> Dict[str, float]:
        """Logs mathematical degradation during the recovery window."""
        latency_spike = latency_after - latency_before
        return {
            "round": float(round_idx),
            "failed_nodes_count": float(len(failed_nodes)),
            "blocked_links_count": float(len(blocked_links)),
            "latency_spike_s": latency_spike,
            "accuracy_degradation": accuracy_drop
        }

if __name__ == "__main__":
    print("Running chaos_injector.py self-tests...")
    
    injector = ChaosInjector(seed=99)
    
    # Test Hardware Failures
    e_res = np.array([1000.0, 500.0, 200.0, 800.0])
    active = [0, 1, 2, 3]
    failed = injector.inject_hardware_failures(e_res, active, p_fail=0.5)
    
    for f in failed:
        assert e_res[f] == 0.0, f"UAV {f} energy should be 0.0"
    
    non_failed = set(active) - set(failed)
    for nf in non_failed:
        assert e_res[nf] > 0.0, f"UAV {nf} energy should be untouched"
        
    print(f"  [OK] Hardware failures injected: {failed}")
    
    # Test Deep Fading
    gains = np.ones((4, 4))
    links = [(0, 1), (1, 2), (2, 3)]
    blocked = injector.inject_deep_fading(gains, links, p_block=0.5, fading_factor=1e-6)
    
    for tx, rx in blocked:
        assert gains[tx, rx] < 1e-5, f"Link ({tx},{rx}) gain not degraded."
        assert gains[rx, tx] < 1e-5, f"Symmetric link ({rx},{tx}) not degraded."
        
    print(f"  [OK] Deep fading injected: {blocked}")
    
    # Test Logging
    log = injector.log_recovery_metrics(10, failed, blocked, 2.5, 4.0, 0.05)
    assert log["latency_spike_s"] == 1.5
    print("  [OK] Recovery metrics logged successfully.")
    print("[PASS] chaos_injector.py — All assertions passed.")
