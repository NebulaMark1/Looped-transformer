"""
Compare results from baseline, lora, and full loop experiments.

Usage:
    python analyze.py --results_dir ./results
"""

import argparse
import json
import os
import glob


def load_results(results_dir: str):
    """Load all result JSON files from the results directory."""
    results = {}
    for path in glob.glob(os.path.join(results_dir, "*_results.json")):
        with open(path) as f:
            data = json.load(f)
        mode = data["config"]["mode"]
        key = mode
        if mode == "lora":
            key += f"_r{data['config']['lora_rank']}"
        results[key] = data
    return results


def print_comparison(results: dict):
    """Print a formatted comparison table."""
    if not results:
        print("No results found.")
        return

    # Header
    print(f"{'Config':<20} {'Total Params':>14} {'Trans Params':>14} {'Best Val PPL':>14} {'Train Time':>12} {'Rel PPL':>10}")
    print("-" * 88)

    # Need baseline for relative comparison
    baseline_ppl = None
    for key, r in results.items():
        if r["config"]["mode"] == "baseline":
            baseline_ppl = r["best_val_ppl"]
            break

    for key in sorted(results.keys()):
        r = results[key]
        p = r["param_counts"]
        ppl = r["best_val_ppl"]
        train_time = r["total_train_time"]
        rel_ppl = f"{(ppl / baseline_ppl - 1) * 100:+.1f}%" if baseline_ppl else "N/A"

        print(f"{key:<20} {p['total']:>14,} {p['transformer']:>14,} {ppl:>14.2f} {train_time:>9.0f}s {rel_ppl:>10}")

    print()

    # Parameter efficiency analysis
    print("Parameter Efficiency Analysis:")
    print("-" * 60)
    for key in sorted(results.keys()):
        if key.startswith("baseline"):
            continue
        r = results[key]
        p = r["param_counts"]
        ppl = r["best_val_ppl"]
        if baseline_ppl:
            param_increase = (p["transformer"] / results["baseline"]["param_counts"]["transformer"] - 1) * 100
            ppl_decrease = (1 - ppl / baseline_ppl) * 100
            print(f"  {key}: {param_increase:+.1f}% params vs baseline, {ppl_decrease:+.1f}% ppl change")

    print()

    # Best epoch for each config
    print("Best Epoch Summary:")
    print("-" * 50)
    for key in sorted(results.keys()):
        r = results[key]
        best = min(r["metrics"], key=lambda x: x["val_ppl"])
        print(f"  {key}: epoch {best['epoch']}, val_ppl={best['val_ppl']:.2f}, train_ppl={best['train_ppl']:.2f}")


def print_per_epoch(results: dict):
    """Print per-epoch metrics for each config."""
    for key in sorted(results.keys()):
        r = results[key]
        print(f"\n{key}:")
        print(f"  {'Epoch':<8} {'Train PPL':<12} {'Val PPL':<12}")
        print(f"  {'-'*32}")
        for m in r["metrics"]:
            print(f"  {m['epoch']:<8} {m['train_ppl']:<12.2f} {m['val_ppl']:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Looped Transformer experiment results")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--detail", action="store_true", help="Show per-epoch breakdown")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in '{args.results_dir}'. Run train.py first.")
        return

    print("\n" + "=" * 88)
    print("  Looped Transformer with Per-Loop LoRA — Experiment Results")
    print("=" * 88 + "\n")

    print_comparison(results)

    if args.detail:
        print_per_epoch(results)

    print()


if __name__ == "__main__":
    main()
