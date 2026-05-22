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
        basename = os.path.basename(path)
        if "oracle_results" in basename or "full_ft_results" in basename:
            continue  # handled separately below
        with open(path) as f:
            data = json.load(f)
        if "config" not in data:
            continue
        cfg = data["config"]
        key = cfg["mode"]
        if cfg["mode"] == "lora":
            key += f"_r{cfg['lora_rank']}"
        if cfg.get("freeze_base"):
            key += "_frozen"
        results[key] = data

    # Also load oracle results if available
    oracle_path = os.path.join(results_dir, "oracle_results.json")
    if os.path.exists(oracle_path):
        results["_oracle"] = {"_type": "oracle", "_path": oracle_path}

    return results


def print_comparison(results: dict, oracle_data: dict | None = None):
    """Print a formatted comparison table."""
    if not results:
        print("No training results found.")
        return

    # Header
    print(f"{'Config':<22} {'Total Params':>14} {'Trans Params':>14} {'Best Val PPL':>14} {'Train Time':>12} {'Rel PPL':>10}")
    print("-" * 90)

    # Find baseline for relative comparison
    baseline_ppl = None
    for key, r in results.items():
        if r.get("config", {}).get("mode") == "baseline":
            baseline_ppl = r["best_val_ppl"]
            break

    for key in sorted(results.keys()):
        r = train_results[key]
        p = r.get("param_counts", {})
        ppl = r["best_val_ppl"]
        train_time = r.get("total_train_time", 0)
        rel_ppl = f"{(ppl / baseline_ppl - 1) * 100:+.1f}%" if baseline_ppl else "N/A"

        print(f"{key:<22} {p.get('total', 0):>14,} {p.get('transformer', 0):>14,} {ppl:>14.2f} {train_time:>9.0f}s {rel_ppl:>10}")

    print()

    # Parameter efficiency analysis
    print("Parameter Efficiency Analysis:")
    print("-" * 60)
    for key in sorted(results.keys()):
        if key.startswith("baseline"):
            continue
        r = train_results[key]
        p = r.get("param_counts", {})
        ppl = r["best_val_ppl"]
        if baseline_ppl and results.get("baseline"):
            param_increase = (p.get("transformer", 0) / results["baseline"]["param_counts"]["transformer"] - 1) * 100
            ppl_decrease = (1 - ppl / baseline_ppl) * 100
            print(f"  {key}: {param_increase:+.1f}% params vs baseline, {ppl_decrease:+.1f}% ppl change")

    print()

    # Best epoch for each config
    print("Best Epoch Summary:")
    print("-" * 50)
    for key in sorted(results.keys()):
        r = train_results[key]
        best = min(r["metrics"], key=lambda x: x["val_ppl"])
        print(f"  {key}: epoch {best['epoch']}, val_ppl={best['val_ppl']:.2f}, train_ppl={best['train_ppl']:.2f}")


def print_per_epoch(results: dict):
    """Print per-epoch metrics for each config (skip oracle)."""
    for key in sorted(results.keys()):
        if key.startswith("_"):
            continue
        r = results[key]
        print(f"\n{key}:")
        print(f"  {'Epoch':<8} {'Train PPL':<12} {'Val PPL':<12}")
        print(f"  {'-'*32}")
        for m in r["metrics"]:
            print(f"  {m['epoch']:<8} {m['train_ppl']:<12.2f} {m['val_ppl']:<12.2f}")


def print_oracle(oracle_data: dict | None, baseline_ppl: float | None):
    """Print oracle LoRA analysis results."""
    if oracle_data is None:
        return
    path = oracle_data.get("_path", "")
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)

    print("\n" + "=" * 72)
    print("  Oracle LoRA — SVD Residual Decomposition")
    print("=" * 72)
    print(f"  Baseline PPL:              {d['baseline_ppl']:.2f}")
    print(f"  Full (upper bound) PPL:    {d.get('full_ppl', 0):.2f}")
    if d.get("trained_lora_ppl"):
        print(f"  Trained LoRA r=8 PPL:      {d['trained_lora_ppl']:.2f}")
    print(f"\n  {'Rank':<8} {'Oracle PPL':<14} {'Recovery %':<14}")
    print(f"  {'-'*38}")
    full_gain = d.get("full_gain", 0)
    for rank_str, ppl in sorted(d["oracle_results"].items(), key=lambda x: int(x[0])):
        rank = int(rank_str)
        delta = d["baseline_ppl"] - ppl
        recovery = (delta / full_gain * 100) if full_gain > 0 else 0
        print(f"  {rank:<8} {ppl:<14.2f} {recovery:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze Looped Transformer experiment results")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--detail", action="store_true", help="Show per-epoch breakdown")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in '{args.results_dir}'. Run train.py first.")
        return

    oracle_data = results.pop("_oracle", None)

    print("\n" + "=" * 88)
    print("  Looped Transformer with Per-Loop LoRA — Experiment Results")
    print("=" * 88 + "\n")

    print_comparison(results)
    print_oracle(oracle_data, None)

    if args.detail:
        print_per_epoch(results)

    print()


if __name__ == "__main__":
    main()
