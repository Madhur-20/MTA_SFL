import json
import os
import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Hotfix for when the root directory is not named "mta_sfl"
if _SCRIPT_DIR.name != "mta_sfl":
    sys.path.insert(0, str(_SCRIPT_DIR))
    sys.modules["mta_sfl"] = sys.modules.get(__name__)

from baselines import FedATCBaseline, SFLSCCBaseline, AdaptSFLBaseline, SHSFLBaseline
from main import run_dataset_experiment

def main():
    dataset = "cifar10"
    episode = 1
    
    os.makedirs("results", exist_ok=True)
    
    results = {}

    print(f"\n{'='*50}\n--- Running Proposed Method: MTA-SFL ---\n{'='*50}")
    # Run the full proposed algorithm
    mta_sfl_stats = run_dataset_experiment(dataset)
    
    # Extract round-by-round lists from the MTA-SFL checkpoint file for plotting
    round_accuracies, round_latencies, round_energies = [], [], []
    checkpoint_path = f"results/results_{dataset}.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            chk = json.load(f)
            rounds = chk.get("rounds", [])
            round_accuracies = [r.get("test_accuracy", 0.0) / 100.0 for r in rounds]
            round_latencies = [r.get("avg_latency_s", 0.0) for r in rounds]
            round_energies = [r.get("cumulative_energy_J", 0.0) for r in rounds]

    results["MTA-SFL (Proposed)"] = {
        "episode": episode,
        "global_accuracy": mta_sfl_stats.get("final_accuracy", 0.0),
        "avg_latency_s": mta_sfl_stats.get("avg_latency_s", 0.0),
        "total_energy_J": round(mta_sfl_stats.get("total_energy_J", 0.0), 2),
        "drones_depleted": mta_sfl_stats.get("drones_depleted", 0),
        "compliance_rate": mta_sfl_stats.get("compliance_rate", 100.0),
        "round_accuracies": round_accuracies,
        "round_latencies": round_latencies,
        "round_energies": round_energies
    }
    print(f"Result for MTA-SFL: {results['MTA-SFL (Proposed)']}")
    
    baselines = {
        "FedATC": FedATCBaseline(),
        "SFLSCC": SFLSCCBaseline(),
        "AdaptSFL": AdaptSFLBaseline(),
        "SHSFL": SHSFLBaseline()
    }
    
    for name, bl_runner in baselines.items():
        print(f"\n{'='*50}\n--- Running Baseline: {name} ---\n{'='*50}")
        ep_metrics = bl_runner.run_episode(dataset, episode)
        results[name] = ep_metrics.summary_dict()
        print(f"Result for {name}: {results[name]}")
        
    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[OK] Baseline runs complete. Results saved to results/baseline_results.json")

if __name__ == "__main__":
    main()

