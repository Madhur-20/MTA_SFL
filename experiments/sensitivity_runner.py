"""MTA-SFL Objective Weight Sensitivity Analysis.

Systematically searches over the global potential function weights (Lambda)
to validate the robustness and interpretability of the reward structure.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pandas as pd


class SensitivityRunner:
    """Iterates over objective weight tuples (λ1, λ2, λ3) and logs shifts."""

    def __init__(self, sim_function: Callable[[float, float, float], Dict[str, float]]) -> None:
        """
        Args:
            sim_function: Callable that accepts (lambda1, lambda2, lambda3) and
                          returns a dictionary of resulting performance metrics.
        """
        self.sim_function = sim_function

    def run_lambda_grid(self) -> pd.DataFrame:
        """Executes a grid search over key Lambda profiles.
        
        Lambda definitions (Eq. 33):
            λ1: Latency weight
            λ2: Energy weight
            λ3: Topology/CC weight
            
        Returns:
            Pandas DataFrame mapping the profiles to resulting physical metrics.
        """
        # Profile Name, λ1, λ2, λ3
        grid: List[Tuple[str, float, float, float]] = [
            ("Latency-Critical", 0.80, 0.10, 0.10),
            ("Energy-Critical",  0.10, 0.80, 0.10),
            ("Topology-Critical",0.10, 0.10, 0.80),
            ("Balanced",         0.33, 0.33, 0.34),
        ]
        
        results: List[Dict[str, float | str]] = []
        for name, l1, l2, l3 in grid:
            assert abs(l1 + l2 + l3 - 1.0) < 1e-6, "Lambdas must sum to 1.0"
            
            res = self.sim_function(l1, l2, l3)
            
            res_dict: Dict[str, float | str] = {
                "Profile": name,
                "Lambda_1_Lat": l1,
                "Lambda_2_Eng": l2,
                "Lambda_3_Top": l3
            }
            res_dict.update(res)
            results.append(res_dict)
            
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running sensitivity_runner.py self-tests...")
    
    # Dummy simulation returning metrics dependent on lambda weights
    def dummy_sim(l1: float, l2: float, l3: float) -> Dict[str, float]:
        # High latency weight means lower actual latency achieved, but costs energy
        latency = 100.0 - (l1 * 50.0) 
        energy = 10000.0 - (l2 * 5000.0)
        cc = 0.5 + (l3 * 0.4)
        
        return {
            "Final_Latency_s": latency,
            "Total_Energy_J": energy,
            "Avg_Clustering_Coef": cc
        }
        
    runner = SensitivityRunner(sim_function=dummy_sim)
    df = runner.run_lambda_grid()
    
    assert len(df) == 4, "Should have exactly 4 profiles."
    
    # Check that Latency-Critical has the lowest latency
    lat_crit = df[df["Profile"] == "Latency-Critical"].iloc[0]
    eng_crit = df[df["Profile"] == "Energy-Critical"].iloc[0]
    
    assert lat_crit["Final_Latency_s"] < eng_crit["Final_Latency_s"]
    assert eng_crit["Total_Energy_J"] < lat_crit["Total_Energy_J"]
    
    print("  [OK] Sensitivity grid executes and captures metric shifts.")
    print("  [OK] Lambda bounds checking enforces sum = 1.0.")
    print(df.to_string())
    print("[PASS] sensitivity_runner.py — All assertions passed.")
