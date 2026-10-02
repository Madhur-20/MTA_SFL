"""Unified episode-level metrics container for baselines and ablation studies.

Provides a standard interface for collecting global accuracy, average latency,
total swarm energy, and drone depletion counts from any experiment run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EpisodeMetrics:
    """Aggregate metrics for one episode (all rounds combined).

    Attributes:
        episode: Episode index.
        global_accuracy: Final test accuracy (0-1 scale).
        avg_latency: Average per-round latency across the episode (s).
        total_energy: Total energy consumed by the swarm (J).
        drones_depleted: Count of UAV nodes whose energy fell below E_min.
        extra: Optional dict for additional metrics.
    """
    episode: int = 0
    global_accuracy: float = 0.0
    avg_latency: float = 0.0
    total_energy: float = 0.0
    drones_depleted: int = 0
    round_accuracies: List[float] = field(default_factory=list)
    round_latencies: List[float] = field(default_factory=list)
    round_energies: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary suitable for JSON serialization."""
        return {
            "episode": self.episode,
            "global_accuracy": round(self.global_accuracy * 100, 2),
            "avg_latency_s": round(self.avg_latency, 4),
            "total_energy_J": round(self.total_energy, 2),
            "drones_depleted": self.drones_depleted,
            "round_accuracies": self.round_accuracies,
            "round_latencies": self.round_latencies,
            "round_energies": self.round_energies,
            **self.extra,
        }
