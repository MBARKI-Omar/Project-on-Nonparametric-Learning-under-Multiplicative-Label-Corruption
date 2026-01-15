"""
Master script to run all experiments sequentially.
"""

import subprocess
import sys
import time


def run_experiment(script_name):
    """Run a single experiment script."""
    print("\n" + "=" * 70)
    print(f"Running {script_name}...")
    print("=" * 70)
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, f'experiments/{script_name}'],
        capture_output=False
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {script_name} FAILED with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ {script_name} completed in {elapsed:.1f}s")
        return True


def main():
    """Run all experiments in sequence."""
    print("\n" + "#" * 70)
    print("# Running All Experiments for Part III")
    print("#" * 70)
    
    experiments = [
        'exp0_basic_validation.py',
        'exp1_bandwidth_selection.py',
        'exp2_sample_size.py',
        'exp3_dimension_effect.py',
        'exp4_boundary_robustness.py',
        'exp5_noise_level_effect.py'
    ]
    
    total_start = time.time()
    results = {}
    
    for exp in experiments:
        success = run_experiment(exp)
        results[exp] = 'SUCCESS' if success else 'FAILED'
        
        if not success:
            print(f"\n⚠️  Stopping due to failure in {exp}")
            break
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "#" * 70)
    print("# Experiment Summary")
    print("#" * 70)
    
    for exp, status in results.items():
        symbol = "✅" if status == 'SUCCESS' else "❌"
        print(f"{symbol} {exp}: {status}")
    
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    if all(s == 'SUCCESS' for s in results.values()):
        print("\n🎉 All experiments completed successfully!")
        print("\nResults available in:")
        print("  - results/*.json")
        print("  - reports/figures/*.pdf")
    else:
        print("\n⚠️  Some experiments failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()