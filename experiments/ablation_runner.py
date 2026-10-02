"""MTA-SFL Component-Wise Ablation Pipeline.

Automates experimental runs toggling DDQN, LinUCB, Top-k sparsification, 
and topological bridging to evaluate isolated component contributions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd


@dataclass
class AblationConfig:
    """Toggles for component-wise ablation."""
    disable_ddqn: bool = False
    disable_linucb: bool = False
    disable_topk: bool = False
    disable_bridge: bool = False


class AblationRunner:
    """Systematically disables components and tracks resulting metrics."""

    def __init__(self, sim_function: Callable[[AblationConfig], Dict[str, float]]) -> None:
        """
        Args:
            sim_function: A callable that runs the simulation for a given AblationConfig
                          and returns a dictionary of converged metrics.
        """
        self.sim_function = sim_function

    def run_all_ablations(self) -> pd.DataFrame:
        """Runs the standard set of ablation configurations.
        
        Returns:
            A Pandas DataFrame mapping config string to converged metrics.
        """
        configs: Dict[str, AblationConfig] = {
            "Full_MTA_SFL": AblationConfig(),
            "No_DDQN_Static_Routing": AblationConfig(disable_ddqn=True),
            "No_LinUCB_Static_Split": AblationConfig(disable_linucb=True),
            "No_TopK_Dense_Comm": AblationConfig(disable_topk=True),
            "No_Bridge_Forced_Compute": AblationConfig(disable_bridge=True),
        }
        
        results: List[Dict[str, float | str]] = []
        for name, cfg in configs.items():
            res = self.sim_function(cfg)
            res_dict: Dict[str, float | str] = {"Configuration": name}
            res_dict.update(res)
            results.append(res_dict)
            
        df = pd.DataFrame(results)
        return df

    def export_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Exports the ablation results to CSV."""
        df.to_csv(filepath, index=False)


if __name__ == "__main__":
    print("Running ablation_runner.py self-tests...")
    
    # Dummy simulation function that returns deterministic outcomes
    def dummy_sim(cfg: AblationConfig) -> Dict[str, float]:
        acc = 0.95
        lat = 100.0
        energy = 5000.0
        
        if cfg.disable_ddqn:
            acc -= 0.10
            energy += 1000.0
        if cfg.disable_linucb:
            acc -= 0.05
            lat += 50.0
        if cfg.disable_topk:
            lat += 200.0
            energy += 2000.0
        if cfg.disable_bridge:
            energy += 1500.0
            
        return {
            "Converged_Accuracy": acc,
            "Cumulative_Latency_s": lat,
            "Total_Swarm_Energy_J": energy
        }
        
    runner = AblationRunner(sim_function=dummy_sim)
    df = runner.run_all_ablations()
    
    assert len(df) == 5, "Should have exactly 5 ablation configs."
    assert "Configuration" in df.columns
    assert "Converged_Accuracy" in df.columns
    
    full_acc = df[df["Configuration"] == "Full_MTA_SFL"]["Converged_Accuracy"].iloc[0]
    no_ddqn_acc = df[df["Configuration"] == "No_DDQN_Static_Routing"]["Converged_Accuracy"].iloc[0]
    
    assert full_acc == 0.95
    assert abs(no_ddqn_acc - 0.85) < 1e-6
    
    print("  [OK] Ablation configurations ran correctly.")
    print("  [OK] DataFrame structure matches expectations.")
    print(df.to_string())
    print("[PASS] ablation_runner.py — All assertions passed.")
