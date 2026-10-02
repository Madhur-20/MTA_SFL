import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_json_file(json_path, output_filename, title_prefix=""):
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    models = list(results.keys())
    has_rounds = all("round_accuracies" in data and len(data["round_accuracies"]) > 0 for data in results.values())
    
    if not has_rounds:
        print(f"Warning: Round data missing in {json_path}")
        return


    # Style configuration to match the provided academic screenshot
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fontsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14
    })
    
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, name in enumerate(models):
        data = results[name]
        # X-axis scaled for every 10 rounds
        rounds = [j * 10 for j in range(1, len(data["round_accuracies"]) + 1)]
        
        # Calculate markevery to ensure roughly 10-15 markers across the line, 
        # mimicking the sparse marker look in the uploaded plot
        mevery = max(1, len(rounds) // 15)
        
        style_kwargs = {
            "label": name,
            "marker": markers[i % len(markers)],
            "markevery": mevery,
            "linewidth": 2.0,
            "markersize": 7,
            "markerfacecolor": "white",
            "markeredgewidth": 1.5
        }
        
        axs[0].plot(rounds, data["round_accuracies"], **style_kwargs)
        axs[1].plot(rounds, data["round_latencies"], **style_kwargs)
        axs[2].plot(rounds, data["round_energies"], **style_kwargs)

    # Apply consistent styling to all subplots
    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.5, color="gray")
        ax.set_xlabel("Round (Epoch)")
        ax.legend(loc="best")

    axs[0].set_title(f"{title_prefix} Test Accuracy")
    axs[0].set_ylabel("Accuracy")
    
    axs[1].set_title(f"{title_prefix} Pipeline Latency")
    axs[1].set_ylabel("Latency (s)")
    
    axs[2].set_title(f"{title_prefix} Cumulative Energy")
    axs[2].set_ylabel("Energy (Joules)")

    plt.tight_layout()
    output_path = os.path.join("results", output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plots successfully generated and saved to {output_path}")

def main():
    os.makedirs("results", exist_ok=True)
    # Plot Baselines
    if os.path.exists("results/baseline_results.json"):
        plot_json_file("results/baseline_results.json", "learning_curves_baselines.png", "Baselines:")
    else:
        print("results/baseline_results.json not found.")
        
    # Plot Ablations
    if os.path.exists("results/ablation_results.json"):
        plot_json_file("results/ablation_results.json", "learning_curves_ablations.png", "Ablations:")
    else:
        print("results/ablation_results.json not found.")

if __name__ == "__main__":
    main()
