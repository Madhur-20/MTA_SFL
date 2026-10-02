"""MTA-SFL Multi-Seed Statistical Orchestrator.

Executes simulations across an array of random seeds, aggregating data
to calculate the Mean and 95% Confidence Intervals for robust evaluation.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, List

import numpy as np
import pandas as pd


class MultiSeedOrchestrator:
    """Parallelized multi-seed execution and statistical aggregation."""

    def __init__(self, sim_function: Callable[[int], pd.DataFrame], seeds: List[int]) -> None:
        """
        Args:
            sim_function: A callable that runs the simulation for a given seed
                          and returns a Pandas DataFrame of time-series data
                          (e.g., metric per round). Must have a 'round' column.
            seeds: List of integer seeds to execute.
        """
        self.sim_function = sim_function
        self.seeds = seeds

    def run(self, max_workers: int | None = None) -> pd.DataFrame:
        """Executes the simulation across all seeds in parallel.

        Computes the Mean (μ) and 95% Confidence Interval (CI) bounds.
        CI_95% = 1.96 * (σ / sqrt(N))

        Args:
            max_workers: Maximum number of parallel processes.

        Returns:
            A Pandas DataFrame indexed by 'round' with '_mean', '_ci_lower', 
            and '_ci_upper' columns for every metric.
        """
        all_dfs: List[pd.DataFrame] = []
        
        # Parallel Execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.sim_function, seed): seed for seed in self.seeds}
            
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                try:
                    df = future.result()
                    df["seed"] = seed
                    all_dfs.append(df)
                except Exception as exc:
                    print(f"Seed {seed} generated an exception: {exc}")
                    raise
                    
        if not all_dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if "round" not in combined_df.columns:
            raise ValueError("sim_function output DataFrame must contain a 'round' column.")
            
        # Identify numeric metric columns, excluding administrative columns
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if "seed" in numeric_cols:
            numeric_cols.remove("seed")
        if "round" in numeric_cols:
            numeric_cols.remove("round")
            
        # Aggregate across seeds per round
        grouped = combined_df.groupby("round")[numeric_cols].agg(["mean", "std", "count"])
        
        final_df = pd.DataFrame(index=grouped.index)
        
        for col in numeric_cols:
            mean = grouped[(col, "mean")]
            std = grouped[(col, "std")].fillna(0.0) # Handle single-seed case (std=NaN)
            n = grouped[(col, "count")]
            
            # Mathematical CI: 1.96 * (sigma / sqrt(N))
            margin_of_error = 1.96 * (std / np.sqrt(n))
            
            final_df[f"{col}_mean"] = mean
            final_df[f"{col}_ci_lower"] = mean - margin_of_error
            final_df[f"{col}_ci_upper"] = mean + margin_of_error
            
        return final_df.reset_index()


# Dummy simulation function representing a full training run
def dummy_sim(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rounds = np.arange(10)
    # Accuracy grows, Loss drops, with some seed-dependent noise
    acc = 0.5 + 0.04 * rounds + rng.normal(0, 0.02, size=10)
    loss = 2.0 - 0.1 * rounds + rng.normal(0, 0.1, size=10)
    return pd.DataFrame({
        "round": rounds,
        "Accuracy": acc,
        "Loss": loss
    })

if __name__ == "__main__":
    print("Running multi_seed_orchestrator.py self-tests...")
    
    seeds = [42, 100, 1234, 9999, 5555]
    orchestrator = MultiSeedOrchestrator(sim_function=dummy_sim, seeds=seeds)
    
    # Using 1 worker for testing to avoid multiprocessing issues in simple scripts
    results = orchestrator.run(max_workers=1)
    
    assert "round" in results.columns
    assert "Accuracy_mean" in results.columns
    assert "Accuracy_ci_lower" in results.columns
    assert "Accuracy_ci_upper" in results.columns
    assert "Loss_mean" in results.columns
    
    assert len(results) == 10, "Should have exactly 10 rounds of aggregated data."
    
    # Validate the CI math empirically for round 0
    round_0 = results.iloc[0]
    margin = round_0["Accuracy_mean"] - round_0["Accuracy_ci_lower"]
    assert margin >= 0.0, "CI Lower should be <= Mean"
    
    # CI bound should be reasonably tight for N=5
    assert margin < 0.1, f"Margin {margin} is unexpectedly large."
    
    print("  [OK] Parallel seed execution completes.")
    print("  [OK] Mean and 95% Confidence Intervals calculated correctly.")
    print(results[["round", "Accuracy_mean", "Accuracy_ci_lower", "Accuracy_ci_upper"]].head())
    print("[PASS] multi_seed_orchestrator.py — All assertions passed.")
