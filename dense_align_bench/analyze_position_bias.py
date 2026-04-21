#!/usr/bin/env python3
"""
DenseAlignBench — Analyze Position Bias in Pairwise Comparison Results

This script checks if the evaluation results are consistent when image order is shuffled
vs. not shuffled. It helps validate that position bias has been successfully mitigated.

For each model pair, it computes:
1. Win rate when images are NOT shuffled (shuffled=false)
2. Win rate when images ARE shuffled (shuffled=true)
3. Statistical test to see if the two distributions are significantly different

A well-designed evaluation should show NO significant difference between shuffled and
non-shuffled results, indicating position bias has been eliminated.

Part of the PromptEcho project:
  "PromptEcho: Annotation-Free Reward from Vision-Language Models
   for Text-to-Image Reinforcement Learning" (arXiv:2604.12652)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_comparison_results(jsonl_file: Path) -> List[Dict]:
    """Load comparison results from JSONL file."""
    results = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    return results


def analyze_position_bias(results: List[Dict], model_a_name: str, model_b_name: str) -> Dict:
    """
    Analyze position bias by comparing shuffled vs non-shuffled results.

    Args:
        results: List of comparison results
        model_a_name: Name of first model (from filename)
        model_b_name: Name of second model (from filename)

    Returns:
        Dict with analysis results
    """
    # Separate results by shuffle status
    shuffled_results = [r for r in results if r.get("shuffled", False)]
    non_shuffled_results = [r for r in results if not r.get("shuffled", False)]

    # Count preferences for non-shuffled samples
    non_shuffled_prefs = Counter()
    for r in non_shuffled_results:
        pref = r["comparison"]["preference"]
        # Map preference to original model names
        if pref == "image_a":
            # When not shuffled, image_a is model_a_original
            winner = r["model_a_original"]
        elif pref == "image_b":
            winner = r["model_b_original"]
        else:
            winner = "tie"
        non_shuffled_prefs[winner] += 1

    # Count preferences for shuffled samples
    shuffled_prefs = Counter()
    for r in shuffled_results:
        pref = r["comparison"]["preference"]
        # When shuffled, need to map back to original models
        if pref == "image_a":
            # image_a after shuffle was originally model_b
            winner = r["model_b_original"]
        elif pref == "image_b":
            # image_b after shuffle was originally model_a
            winner = r["model_a_original"]
        else:
            winner = "tie"
        shuffled_prefs[winner] += 1

    # Calculate win rates
    total_non_shuffled = len(non_shuffled_results)
    total_shuffled = len(shuffled_results)

    non_shuffled_stats = {
        "total": total_non_shuffled,
        model_a_name: non_shuffled_prefs.get(model_a_name, 0),
        model_b_name: non_shuffled_prefs.get(model_b_name, 0),
        "tie": non_shuffled_prefs.get("tie", 0),
        f"{model_a_name}_rate": non_shuffled_prefs.get(model_a_name, 0) / total_non_shuffled * 100 if total_non_shuffled > 0 else 0,
        f"{model_b_name}_rate": non_shuffled_prefs.get(model_b_name, 0) / total_non_shuffled * 100 if total_non_shuffled > 0 else 0,
        "tie_rate": non_shuffled_prefs.get("tie", 0) / total_non_shuffled * 100 if total_non_shuffled > 0 else 0,
    }

    shuffled_stats = {
        "total": total_shuffled,
        model_a_name: shuffled_prefs.get(model_a_name, 0),
        model_b_name: shuffled_prefs.get(model_b_name, 0),
        "tie": shuffled_prefs.get("tie", 0),
        f"{model_a_name}_rate": shuffled_prefs.get(model_a_name, 0) / total_shuffled * 100 if total_shuffled > 0 else 0,
        f"{model_b_name}_rate": shuffled_prefs.get(model_b_name, 0) / total_shuffled * 100 if total_shuffled > 0 else 0,
        "tie_rate": shuffled_prefs.get("tie", 0) / total_shuffled * 100 if total_shuffled > 0 else 0,
    }

    # Statistical test: Chi-square test for independence
    # H0: Shuffle status and winner are independent (no position bias)
    contingency_table = [
        [non_shuffled_prefs.get(model_a_name, 0),
         non_shuffled_prefs.get(model_b_name, 0),
         non_shuffled_prefs.get("tie", 0)],
        [shuffled_prefs.get(model_a_name, 0),
         shuffled_prefs.get(model_b_name, 0),
         shuffled_prefs.get("tie", 0)]
    ]

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Interpretation
    if p_value < 0.01:
        interpretation = "SIGNIFICANT position bias detected (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Possible position bias (p < 0.05)"
    else:
        interpretation = "No significant position bias detected"

    return {
        "non_shuffled": non_shuffled_stats,
        "shuffled": shuffled_stats,
        "chi2": chi2,
        "p_value": p_value,
        "interpretation": interpretation
    }


def visualize_position_bias(analysis_results: Dict, model_a_name: str, model_b_name: str, output_file: Path):
    """Create visualization comparing shuffled vs non-shuffled results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data for plotting
    categories = [model_a_name, model_b_name, "Tie"]
    non_shuffled_rates = [
        analysis_results["non_shuffled"][f"{model_a_name}_rate"],
        analysis_results["non_shuffled"][f"{model_b_name}_rate"],
        analysis_results["non_shuffled"]["tie_rate"]
    ]
    shuffled_rates = [
        analysis_results["shuffled"][f"{model_a_name}_rate"],
        analysis_results["shuffled"][f"{model_b_name}_rate"],
        analysis_results["shuffled"]["tie_rate"]
    ]

    # Plot 1: Side-by-side bars
    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width/2, non_shuffled_rates, width, label='Not Shuffled', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, shuffled_rates, width, label='Shuffled', alpha=0.8, color='coral')

    axes[0].set_xlabel('Winner', fontsize=12)
    axes[0].set_ylabel('Win Rate (%)', fontsize=12)
    axes[0].set_title(f'Position Bias Analysis\n{model_a_name} vs {model_b_name}', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (ns, s) in enumerate(zip(non_shuffled_rates, shuffled_rates)):
        axes[0].text(i - width/2, ns + 1, f'{ns:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, s + 1, f'{s:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Difference plot
    differences = [s - ns for ns, s in zip(non_shuffled_rates, shuffled_rates)]
    colors = ['green' if abs(d) < 5 else 'orange' if abs(d) < 10 else 'red' for d in differences]

    axes[1].bar(categories, differences, alpha=0.8, color=colors)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Winner', fontsize=12)
    axes[1].set_ylabel('Difference (Shuffled - Not Shuffled) %', fontsize=12)
    axes[1].set_title('Position Bias Magnitude\n(smaller is better)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, d in enumerate(differences):
        axes[1].text(i, d + (0.5 if d >= 0 else -0.5), f'{d:+.1f}%', ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)

    # Add statistical test result
    p_value = analysis_results["p_value"]
    interpretation = analysis_results["interpretation"]
    fig.text(0.5, 0.02, f'Chi-square test: p = {p_value:.4f} -- {interpretation}',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()


def generate_report(all_analyses: Dict[str, Dict], output_file: Path):
    """Generate markdown report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Position Bias Analysis Report\n\n")
        f.write("This report analyzes whether the random shuffling successfully eliminated position bias.\n\n")
        f.write("## Methodology\n\n")
        f.write("For each model pair, we compare:\n")
        f.write("- **Non-shuffled samples** (shuffled=false): Original image order (model_a in position A, model_b in position B)\n")
        f.write("- **Shuffled samples** (shuffled=true): Swapped image order (model_b in position A, model_a in position B)\n\n")
        f.write("If position bias exists, we would see different win rates between shuffled and non-shuffled groups.\n\n")
        f.write("## Results\n\n")

        for pair_name, analysis in all_analyses.items():
            f.write(f"### {pair_name}\n\n")

            # Table
            f.write("| Metric | Not Shuffled | Shuffled | Difference |\n")
            f.write("|--------|--------------|----------|------------|\n")

            model_a_name = [k for k in analysis["non_shuffled"].keys() if k.endswith("_rate")][0].replace("_rate", "")
            model_b_name = [k for k in analysis["non_shuffled"].keys() if k.endswith("_rate")][1].replace("_rate", "")

            ns = analysis["non_shuffled"]
            sh = analysis["shuffled"]

            f.write(f"| Sample count | {ns['total']} | {sh['total']} | - |\n")
            f.write(f"| {model_a_name} win rate | {ns[f'{model_a_name}_rate']:.2f}% | {sh[f'{model_a_name}_rate']:.2f}% | {sh[f'{model_a_name}_rate'] - ns[f'{model_a_name}_rate']:+.2f}% |\n")
            f.write(f"| {model_b_name} win rate | {ns[f'{model_b_name}_rate']:.2f}% | {sh[f'{model_b_name}_rate']:.2f}% | {sh[f'{model_b_name}_rate'] - ns[f'{model_b_name}_rate']:+.2f}% |\n")
            f.write(f"| Tie rate | {ns['tie_rate']:.2f}% | {sh['tie_rate']:.2f}% | {sh['tie_rate'] - ns['tie_rate']:+.2f}% |\n\n")

            # Statistical test
            f.write(f"**Statistical Test (Chi-square):**\n")
            f.write(f"- chi2 = {analysis['chi2']:.4f}\n")
            f.write(f"- p-value = {analysis['p_value']:.4f}\n")
            f.write(f"- **Interpretation:** {analysis['interpretation']}\n\n")

        f.write("## Conclusion\n\n")
        f.write("If all p-values > 0.05, position bias has been successfully mitigated. ")
        f.write("The random shuffling ensures that the evaluation results are independent of image position.\n")


def main():
    parser = argparse.ArgumentParser(
        description="DenseAlignBench — Analyze position bias in pairwise comparison results"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing pairwise comparison JSONL files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: same as input_dir)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DenseAlignBench — Position Bias Analysis")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Find all pairwise comparison files
    comparison_files = list(input_dir.glob("pairwise_comparison_*.jsonl"))
    if not comparison_files:
        print(f"Error: No pairwise comparison files found in {input_dir}")
        return

    print(f"\nFound {len(comparison_files)} comparison files:")
    for f in comparison_files:
        print(f"  - {f.name}")

    all_analyses = {}

    # Analyze each file
    for jsonl_file in comparison_files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {jsonl_file.name}")
        print(f"{'='*80}")

        # Extract model names from filename
        # Format: pairwise_comparison_modelA_vs_modelB.jsonl
        filename = jsonl_file.stem
        parts = filename.replace("pairwise_comparison_", "").split("_vs_")
        if len(parts) != 2:
            print(f"Warning: Cannot parse model names from {jsonl_file.name}")
            continue

        model_a_name, model_b_name = parts

        # Load results
        results = load_comparison_results(jsonl_file)
        print(f"Loaded {len(results)} comparisons")

        # Analyze position bias
        analysis = analyze_position_bias(results, model_a_name, model_b_name)
        all_analyses[f"{model_a_name} vs {model_b_name}"] = analysis

        # Print summary
        print(f"\nNot Shuffled (n={analysis['non_shuffled']['total']}):")
        print(f"  {model_a_name}: {analysis['non_shuffled'][f'{model_a_name}_rate']:.2f}%")
        print(f"  {model_b_name}: {analysis['non_shuffled'][f'{model_b_name}_rate']:.2f}%")
        print(f"  Tie: {analysis['non_shuffled']['tie_rate']:.2f}%")

        print(f"\nShuffled (n={analysis['shuffled']['total']}):")
        print(f"  {model_a_name}: {analysis['shuffled'][f'{model_a_name}_rate']:.2f}%")
        print(f"  {model_b_name}: {analysis['shuffled'][f'{model_b_name}_rate']:.2f}%")
        print(f"  Tie: {analysis['shuffled']['tie_rate']:.2f}%")

        print(f"\nStatistical Test:")
        print(f"  Chi-square: {analysis['chi2']:.4f}")
        print(f"  P-value: {analysis['p_value']:.4f}")
        print(f"  {analysis['interpretation']}")

        # Create visualization
        viz_file = output_dir / f"position_bias_{model_a_name}_vs_{model_b_name}.png"
        visualize_position_bias(analysis, model_a_name, model_b_name, viz_file)

    # Generate report
    report_file = output_dir / "position_bias_analysis_report.md"
    generate_report(all_analyses, report_file)
    print(f"\nReport saved to {report_file}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
