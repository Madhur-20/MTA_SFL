import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if _SCRIPT_DIR.name != "mta_sfl":
    sys.path.insert(0, str(_SCRIPT_DIR))
    sys.modules["mta_sfl"] = sys.modules.get(__name__)

from ablation_studies import STANDARD_ABLATIONS
from main import run_dataset_experiment

def main():
    dataset = "cifar10"
    results = {}
    
    os.makedirs("results", exist_ok=True)
    
    for flags in STANDARD_ABLATIONS:
        print(f"\n{'='*50}\n--- Running Ablation: {flags.name} ---\n{'='*50}")
        stats = run_dataset_experiment(dataset, ablation_flags=flags)
        # Extract round arrays from the checkpoint
        round_accuracies, round_latencies, round_energies = [], [], []
        checkpoint_path = f"results/results_{dataset}.json"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                chk = json.load(f)
                rounds = chk.get("rounds", [])
                round_accuracies = [r.get("test_accuracy", 0.0) / 100.0 for r in rounds]
                round_latencies = [r.get("avg_latency_s", 0.0) for r in rounds]
                round_energies = [r.get("cumulative_energy_J", 0.0) for r in rounds]
        
        results[flags.name] = {
            "global_accuracy": stats.get("final_accuracy", 0.0),
            "avg_latency_s": stats.get("avg_latency_s", 0.0),
            "total_energy_J": round(stats.get("total_energy_J", 0.0), 2),
            "drones_depleted": stats.get("drones_depleted", 0),
            "compliance_rate": stats.get("compliance_rate", 100.0),
            "round_accuracies": round_accuracies,
            "round_latencies": round_latencies,
            "round_energies": round_energies
        }
        print(f"Result for {flags.name}: {results[flags.name]}")
    
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[OK] Ablation runs complete. Results saved to results/ablation_results.json")

if __name__ == "__main__":
    main()

