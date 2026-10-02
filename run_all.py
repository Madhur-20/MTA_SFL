import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {script_name}")
    print(f"{'='*60}")
    
    # Run the script and stream the output directly to the terminal
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}.")
        print("Stopping full execution pipeline.")
        sys.exit(1)
        
    print(f"✅ FINISHED: {script_name}\n")

def main():
    start_time = time.time()
    
    # The order of execution
    scripts = [
        "run_baselines.py",
        "run_ablations.py",
        "plot_results.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    elapsed_hours = (time.time() - start_time) / 3600
    print(f"🎉 ALL EXPERIMENTS AND PLOTTING COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total pipeline execution time: {elapsed_hours:.2f} hours.")
    print(f"📊 Check the 'results/' folder for your JSON files and PNG plots!")

if __name__ == "__main__":
    main()
