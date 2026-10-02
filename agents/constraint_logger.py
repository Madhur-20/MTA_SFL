"""MTA-SFL Constraint & Reviewer Telemetry Logger.

Instruments DDQN and LinUCB agents to log per-round telemetry for:
- Constraint compliance rate
- Attempted violations masked by dynamic action masking
- Exploration vs exploitation ratio (r_int vs r_ext)
- Topological bridge activation frequency
- Full round-level decision records

References
----------
  Reviewer metric:  Constraint compliance tracking per paper Sec. VI.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. Round Record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RoundRecord:
    """Structured telemetry record for one decision round.

    Attributes:
        round_idx: Global round index.
        primary_uav_idx: Index of the primary UAV making the decision.
        selected_relay: Selected relay index (-1 if routing failure).
        selected_split: Selected split configuration (p1, p2, p3).
        ddqn_feasible_count: Number of feasible relays after masking.
        linucb_feasible_count: Number of feasible splits after masking.
        ddqn_attempted_violations: Relays filtered by DDQN mask.
        linucb_attempted_violations: Splits filtered by LinUCB mask.
        constraint_compliant: True if no Ψ penalty applied.
        r_ext: Extrinsic reward.
        r_int: Intrinsic (exploration) reward.
        r_total: Total reward (r_ext + r_int).
        is_bridge_mode: True if p2 == p3.
        bridge_forced: True if bridge was forced by energy masking.
        epsilon: Current DDQN ε for exploration.
        T_total: Actual pipeline latency (s).
        E_ratio: Energy consumption ratio.
        relay_battery_fraction: Relay E_res / E_init.
    """
    round_idx: int = 0
    primary_uav_idx: int = 0
    selected_relay: int = -1
    selected_split: Tuple[int, int, int] = (1, 1, 1)
    ddqn_feasible_count: int = 0
    linucb_feasible_count: int = 0
    ddqn_attempted_violations: int = 0
    linucb_attempted_violations: int = 0
    constraint_compliant: bool = True
    r_ext: float = 0.0
    r_int: float = 0.0
    r_total: float = 0.0
    is_bridge_mode: bool = False
    bridge_forced: bool = False
    epsilon: float = 1.0
    T_total: float = 0.0
    E_ratio: float = 0.0
    relay_battery_fraction: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Constraint Logger
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintLogger:
    """Telemetry logger for constraint compliance and agent decisions.

    Tracks per-round records and provides aggregate statistics for
    analysis and reviewer metrics.
    """

    def __init__(self) -> None:
        self._records: List[RoundRecord] = []

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------

    def log_round(self, record: RoundRecord) -> None:
        """Append a round record to the log.

        Args:
            record: Completed round record.
        """
        self._records.append(record)

    # ----------------------------------------------------------------
    # Constraint Compliance
    # ----------------------------------------------------------------

    def get_compliance_rate(self, window: Optional[int] = None) -> float:
        """Fraction of rounds where agents were constraint-compliant.

        Args:
            window: If set, compute over the last ``window`` rounds.

        Returns:
            Compliance rate ∈ [0, 1]. Returns 0.0 if no records.
        """
        records = self._get_window(window)
        if not records:
            return 0.0
        compliant = sum(1 for r in records if r.constraint_compliant)
        return compliant / len(records)

    # ----------------------------------------------------------------
    # Attempted Violations
    # ----------------------------------------------------------------

    def get_total_attempted_violations(self) -> Dict[str, int]:
        """Total attempted violations prevented by dynamic masking.

        Returns:
            Dict with keys ``'ddqn'`` and ``'linucb'``.
        """
        ddqn_total: int = sum(r.ddqn_attempted_violations for r in self._records)
        linucb_total: int = sum(r.linucb_attempted_violations for r in self._records)
        return {"ddqn": ddqn_total, "linucb": linucb_total}

    # ----------------------------------------------------------------
    # Exploration vs Exploitation
    # ----------------------------------------------------------------

    def get_exploration_exploitation_ratio(
        self, window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Average intrinsic vs extrinsic reward over a window.

        Args:
            window: If set, compute over the last ``window`` rounds.

        Returns:
            Dict with keys ``'avg_r_int'``, ``'avg_r_ext'``, ``'ratio'``.
        """
        records = self._get_window(window)
        if not records:
            return {"avg_r_int": 0.0, "avg_r_ext": 0.0, "ratio": 0.0}

        avg_r_int = sum(r.r_int for r in records) / len(records)
        avg_r_ext = sum(r.r_ext for r in records) / len(records)
        ratio = abs(avg_r_int) / (abs(avg_r_ext) + 1e-8)

        return {
            "avg_r_int": avg_r_int,
            "avg_r_ext": avg_r_ext,
            "ratio": ratio,
        }

    # ----------------------------------------------------------------
    # Bridge Activations
    # ----------------------------------------------------------------

    def get_bridge_activation_rate(
        self, window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Bridge mode activation rates.

        Args:
            window: If set, compute over the last ``window`` rounds.

        Returns:
            Dict with ``'bridge_rate'`` and ``'forced_bridge_rate'``.
        """
        records = self._get_window(window)
        if not records:
            return {"bridge_rate": 0.0, "forced_bridge_rate": 0.0}

        bridge_count = sum(1 for r in records if r.is_bridge_mode)
        forced_count = sum(1 for r in records if r.bridge_forced)

        return {
            "bridge_rate": bridge_count / len(records),
            "forced_bridge_rate": forced_count / len(records),
        }

    # ----------------------------------------------------------------
    # History Queries
    # ----------------------------------------------------------------

    def get_epsilon_history(self) -> List[float]:
        """Return the full epsilon history."""
        return [r.epsilon for r in self._records]

    def get_reward_history(self) -> Dict[str, List[float]]:
        """Return full reward histories.

        Returns:
            Dict with keys ``'r_ext'``, ``'r_int'``, ``'r_total'``.
        """
        return {
            "r_ext": [r.r_ext for r in self._records],
            "r_int": [r.r_int for r in self._records],
            "r_total": [r.r_total for r in self._records],
        }

    # ----------------------------------------------------------------
    # Sliding Window Statistics
    # ----------------------------------------------------------------

    def get_sliding_window_stats(
        self, window: int = 50,
    ) -> Dict[str, List[float]]:
        """Compute sliding window moving averages for plotting.

        Args:
            window: Window size.

        Returns:
            Dict with keys:
            - ``'compliance_rate'``: Moving compliance rate.
            - ``'avg_r_total'``: Moving average total reward.
            - ``'avg_epsilon'``: Moving average epsilon.
            - ``'bridge_rate'``: Moving bridge activation rate.
        """
        n = len(self._records)
        compliance: List[float] = []
        avg_reward: List[float] = []
        avg_eps: List[float] = []
        bridge_rate: List[float] = []

        for i in range(n):
            start = max(0, i - window + 1)
            window_records = self._records[start:i + 1]
            w = len(window_records)

            compliance.append(
                sum(1 for r in window_records if r.constraint_compliant) / w
            )
            avg_reward.append(sum(r.r_total for r in window_records) / w)
            avg_eps.append(sum(r.epsilon for r in window_records) / w)
            bridge_rate.append(
                sum(1 for r in window_records if r.is_bridge_mode) / w
            )

        return {
            "compliance_rate": compliance,
            "avg_r_total": avg_reward,
            "avg_epsilon": avg_eps,
            "bridge_rate": bridge_rate,
        }

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Comprehensive summary of all logged telemetry.

        Returns:
            Dict with aggregate statistics.
        """
        n = len(self._records)
        if n == 0:
            return {"total_rounds": 0}

        violations = self.get_total_attempted_violations()
        ee_ratio = self.get_exploration_exploitation_ratio()
        bridge = self.get_bridge_activation_rate()

        rewards = self.get_reward_history()
        avg_r_total = sum(rewards["r_total"]) / n if n > 0 else 0.0
        avg_T = sum(r.T_total for r in self._records) / n
        avg_E = sum(r.E_ratio for r in self._records) / n

        routing_failures = sum(1 for r in self._records if r.selected_relay == -1)

        return {
            "total_rounds": n,
            "compliance_rate": self.get_compliance_rate(),
            "routing_failures": routing_failures,
            "routing_failure_rate": routing_failures / n,
            "total_ddqn_violations": violations["ddqn"],
            "total_linucb_violations": violations["linucb"],
            "avg_r_total": avg_r_total,
            "avg_r_ext": ee_ratio["avg_r_ext"],
            "avg_r_int": ee_ratio["avg_r_int"],
            "exploration_ratio": ee_ratio["ratio"],
            "bridge_rate": bridge["bridge_rate"],
            "forced_bridge_rate": bridge["forced_bridge_rate"],
            "avg_latency": avg_T,
            "avg_energy_ratio": avg_E,
            "final_epsilon": self._records[-1].epsilon if n > 0 else 1.0,
        }

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def to_dataframe(self) -> Any:
        """Export records to a pandas DataFrame.

        Returns:
            pandas DataFrame if pandas is available, else a list of dicts.
        """
        records_dicts = [asdict(r) for r in self._records]
        try:
            import pandas as pd
            return pd.DataFrame(records_dicts)
        except ImportError:
            return records_dicts

    def export_json(self, filepath: str) -> None:
        """Export all records to a JSON file.

        Args:
            filepath: Destination file path.
        """
        records_dicts = []
        for r in self._records:
            d = asdict(r)
            # Convert tuple to list for JSON serialization
            d["selected_split"] = list(d["selected_split"])
            records_dicts.append(d)

        with open(filepath, "w") as f:
            json.dump(records_dicts, f, indent=2)

    # ----------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded data."""
        self._records.clear()

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_window(self, window: Optional[int]) -> List[RoundRecord]:
        """Return the last ``window`` records, or all if window is None."""
        if window is None or window >= len(self._records):
            return self._records
        return self._records[-window:]

    @property
    def num_records(self) -> int:
        """Total number of logged records."""
        return len(self._records)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import tempfile

    print("Running constraint_logger.py self-tests …")

    logger = ConstraintLogger()

    # ---- Generate 200 synthetic rounds with improving compliance ----
    for i in range(200):
        # Compliance improves over time
        compliant = (i / 200.0) > 0.3 or i % 3 != 0
        epsilon = max(0.05, 1.0 * (0.995 ** i))
        r_ext = -0.5 + 0.003 * i + 0.1 * (1 if compliant else -1)
        r_int = 0.1 / math.sqrt(i + 1)
        is_bridge = i > 150 and i % 5 == 0
        bridge_forced = is_bridge and i > 170

        record = RoundRecord(
            round_idx=i,
            primary_uav_idx=i % 4 + 2,
            selected_relay=3 if compliant else -1,
            selected_split=(3, 6, 6) if is_bridge else (3, 6, 9),
            ddqn_feasible_count=max(1, 4 - i // 60),
            linucb_feasible_count=max(10, 100 - i),
            ddqn_attempted_violations=max(0, 3 - i // 50),
            linucb_attempted_violations=max(0, 50 - i // 2),
            constraint_compliant=compliant,
            r_ext=r_ext,
            r_int=r_int,
            r_total=r_ext + r_int,
            is_bridge_mode=is_bridge,
            bridge_forced=bridge_forced,
            epsilon=epsilon,
            T_total=5.0 - 0.01 * i,
            E_ratio=0.1 + 0.001 * i,
            relay_battery_fraction=max(0.1, 1.0 - 0.004 * i),
        )
        logger.log_round(record)

    assert logger.num_records == 200
    print(f"  [OK] Logged {logger.num_records} rounds")

    # ---- Test compliance rate ----
    cr_all = logger.get_compliance_rate()
    assert 0.0 <= cr_all <= 1.0
    print(f"  [OK] Overall compliance rate: {cr_all:.4f}")

    cr_50 = logger.get_compliance_rate(window=50)
    assert 0.0 <= cr_50 <= 1.0
    print(f"  [OK] Last-50 compliance rate: {cr_50:.4f}")

    # ---- Test violations ----
    viols = logger.get_total_attempted_violations()
    assert viols["ddqn"] >= 0
    assert viols["linucb"] >= 0
    print(f"  [OK] Attempted violations: DDQN={viols['ddqn']}, LinUCB={viols['linucb']}")

    # ---- Test exploration/exploitation ----
    ee = logger.get_exploration_exploitation_ratio()
    assert "avg_r_int" in ee and "avg_r_ext" in ee and "ratio" in ee
    print(f"  [OK] Exploration ratio: {ee['ratio']:.4f}")

    ee_50 = logger.get_exploration_exploitation_ratio(window=50)
    assert ee_50["ratio"] < ee["ratio"], (
        "Recent exploration should be lower (epsilon decayed)"
    )
    print(f"  [OK] Recent exploration ratio ({ee_50['ratio']:.4f}) < overall ({ee['ratio']:.4f})")

    # ---- Test bridge activations ----
    bridge = logger.get_bridge_activation_rate()
    assert 0.0 <= bridge["bridge_rate"] <= 1.0
    assert bridge["forced_bridge_rate"] <= bridge["bridge_rate"]
    print(f"  [OK] Bridge rate: {bridge['bridge_rate']:.4f}, "
          f"forced: {bridge['forced_bridge_rate']:.4f}")

    # ---- Test epsilon history ----
    eps_hist = logger.get_epsilon_history()
    assert len(eps_hist) == 200
    assert eps_hist[0] > eps_hist[-1], "Epsilon should decay"
    print(f"  [OK] Epsilon: {eps_hist[0]:.4f} → {eps_hist[-1]:.4f}")

    # ---- Test reward history ----
    rew_hist = logger.get_reward_history()
    assert len(rew_hist["r_ext"]) == 200
    assert len(rew_hist["r_int"]) == 200
    assert len(rew_hist["r_total"]) == 200
    print("  [OK] Reward histories have 200 entries")

    # ---- Test sliding window ----
    sw = logger.get_sliding_window_stats(window=50)
    assert len(sw["compliance_rate"]) == 200
    assert len(sw["avg_r_total"]) == 200
    print("  [OK] Sliding window stats computed")

    # ---- Test summary ----
    summary = logger.get_summary()
    assert summary["total_rounds"] == 200
    assert "compliance_rate" in summary
    assert "routing_failures" in summary
    assert "exploration_ratio" in summary
    print(f"  [OK] Summary: {json.dumps(summary, indent=2)[:200]}…")

    # ---- Test JSON export ----
    tmpdir = tempfile.mkdtemp()
    json_path = os.path.join(tmpdir, "telemetry.json")
    logger.export_json(json_path)

    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert len(loaded) == 200, f"Expected 200 records, got {len(loaded)}"
    assert loaded[0]["round_idx"] == 0
    assert loaded[-1]["round_idx"] == 199
    print(f"  [OK] JSON export: {json_path} ({len(loaded)} records)")

    # Clean up
    os.remove(json_path)
    os.rmdir(tmpdir)

    # ---- Test DataFrame export ----
    df_result = logger.to_dataframe()
    try:
        import pandas as pd
        assert isinstance(df_result, pd.DataFrame)
        assert len(df_result) == 200
        print("  [OK] DataFrame export (pandas available)")
    except ImportError:
        assert isinstance(df_result, list)
        assert len(df_result) == 200
        print("  [OK] DataFrame export (pandas unavailable, returned list)")

    # ---- Test reset ----
    logger.reset()
    assert logger.num_records == 0
    assert logger.get_compliance_rate() == 0.0
    print("  [OK] Reset clears all records")

    print("\n[PASS] constraint_logger.py — All assertions passed.")
