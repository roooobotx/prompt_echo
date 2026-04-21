#!/usr/bin/env python3
"""
DenseAlignBench — Pairwise Preference Evaluation for Image Generation Models

This script compares images from different models using pairwise preference
evaluation, focusing ONLY on prompt-following accuracy. It supports:
- Random image order shuffling to avoid position bias
- Parallel processing with configurable workers
- Resumable evaluation with --skip_existing
- All pairwise combinations of models or selective pairs

Part of the PromptEcho project:
  "PromptEcho: Annotation-Free Reward from Vision-Language Models
   for Text-to-Image Reinforcement Learning" (arXiv:2604.12652)

Usage:
    python -m dense_align_bench.evaluate_pairwise \
        --input_dir /path/to/inference_results \
        --output_dir ./pairwise_results \
        --model_pairs "all" \
        --num_workers 16
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from PIL import Image
from tqdm import tqdm

try:
    from .evaluator import DenseAlignEvaluator
except ImportError:
    # Allow standalone execution
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluator import DenseAlignEvaluator


def load_image(image_path: Path) -> Optional[Image.Image]:
    """Load image from file path."""
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None


def load_model_metadata(input_dir: Path, model_names: List[str]) -> Dict[str, Dict[str, Dict]]:
    """
    Load metadata from all model directories.

    Args:
        input_dir: Base directory containing model subdirectories
        model_names: List of model directory names

    Returns:
        Dict mapping: {model_name: {sample_id: metadata_dict}}
    """
    metadata_by_model = {}

    for model in model_names:
        model_dir = input_dir / model
        metadata_file = model_dir / "metadata.jsonl"

        if not metadata_file.exists():
            print(f"Warning: metadata.jsonl not found for model {model}")
            continue

        metadata_dict = {}
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Support both "data_id" and "index" as identifier fields
                    sample_id = str(data.get("data_id", data.get("index")))
                    if sample_id is None:
                        print(f"Warning: No 'data_id' or 'index' field in line {line_idx} of {metadata_file}")
                        continue
                    metadata_dict[sample_id] = data
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse line {line_idx} in {metadata_file}: {e}")
                    continue

        metadata_by_model[model] = metadata_dict
        print(f"  Loaded {len(metadata_dict)} samples from {model}")

    return metadata_by_model


def find_common_samples(metadata_by_model: Dict[str, Dict[str, Dict]]) -> List[str]:
    """
    Find common sample IDs across all models.

    Args:
        metadata_by_model: Dict mapping model_name to {sample_id: metadata}

    Returns:
        Sorted list of common sample IDs
    """
    if not metadata_by_model:
        return []

    all_data_ids = [set(m.keys()) for m in metadata_by_model.values()]
    common = set.intersection(*all_data_ids)

    return sorted(list(common))


def compare_single_sample(
    data_id: str,
    model_a_original: str,
    model_b_original: str,
    metadata_by_model: Dict[str, Dict[str, Dict]],
    base_dirs: Dict[str, Path],
    evaluator: DenseAlignEvaluator,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Compare images from two models for a single sample.

    Args:
        data_id: Sample identifier
        model_a_original: First model name (before shuffling)
        model_b_original: Second model name (before shuffling)
        metadata_by_model: Metadata dict for all models
        base_dirs: Base directories for each model
        evaluator: DenseAlignEvaluator instance
        verbose: Whether to print detailed progress

    Returns:
        Dict with comparison results or None if failed
    """
    try:
        # Load metadata
        meta_a = metadata_by_model[model_a_original][data_id]
        meta_b = metadata_by_model[model_b_original][data_id]

        # Load images
        image_path_a = base_dirs[model_a_original] / meta_a["filename"]
        image_path_b = base_dirs[model_b_original] / meta_b["filename"]

        if not image_path_a.exists():
            if verbose:
                print(f"Image not found: {image_path_a}")
            return None

        if not image_path_b.exists():
            if verbose:
                print(f"Image not found: {image_path_b}")
            return None

        image_a = load_image(image_path_a)
        image_b = load_image(image_path_b)

        if image_a is None or image_b is None:
            return None

        # CRITICAL: Random shuffle (50/50) to mitigate position bias
        shuffled = random.random() < 0.5
        if shuffled:
            model_a, model_b = model_b_original, model_a_original
            image_a, image_b = image_b, image_a
            filename_a, filename_b = meta_b["filename"], meta_a["filename"]
        else:
            model_a, model_b = model_a_original, model_b_original
            filename_a, filename_b = meta_a["filename"], meta_b["filename"]

        # Call evaluator with potentially shuffled images
        result = evaluator.compare_two_images_prompt_following(
            prompt=meta_a["caption"],
            image_a=image_a,
            image_b=image_b,
            model_a_name=model_a,
            model_b_name=model_b
        )

        if result is None:
            if verbose:
                print(f"Comparison failed for data_id {data_id}")
            return None

        # Return with original model tracking
        output = {
            "data_id": data_id,
            "caption": meta_a["caption"],
            "model_a": model_a,
            "model_b": model_b,
            "model_a_original": model_a_original,
            "model_b_original": model_b_original,
            "model_a_filename": filename_a,
            "model_b_filename": filename_b,
            "shuffled": shuffled,
            "comparison": {
                "reasoning": result["reasoning"],
                "preference": result["preference"]
            },
            "timestamp": datetime.now().isoformat()
        }

        if verbose:
            pref = result["preference"]
            print(f"  {data_id}: {model_a_original} vs {model_b_original} -> {pref} (shuffled={shuffled})")

        return output

    except Exception as e:
        if verbose:
            print(f"Error comparing {data_id}: {e}")
        return None


def evaluate_model_pair(
    model_a: str,
    model_b: str,
    common_data_ids: List[str],
    metadata_by_model: Dict[str, Dict[str, Dict]],
    base_dirs: Dict[str, Path],
    evaluator: DenseAlignEvaluator,
    output_file: Path,
    num_workers: int = 16,
    skip_existing: bool = False
) -> Dict[str, int]:
    """
    Evaluate all comparisons for one model pair.

    Args:
        model_a: First model name
        model_b: Second model name
        common_data_ids: List of common sample IDs
        metadata_by_model: Metadata dict for all models
        base_dirs: Base directories for each model
        evaluator: DenseAlignEvaluator instance
        output_file: Path to save results
        num_workers: Number of parallel workers
        skip_existing: Skip already evaluated samples

    Returns:
        Dict with statistics (total, success, failed)
    """
    # Load existing results if skip_existing
    existing_data_ids = set()
    if skip_existing and output_file.exists():
        print(f"  Loading existing results from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing_data_ids.add(data["data_id"])
                except Exception:
                    pass
        print(f"  Found {len(existing_data_ids)} existing comparisons")

    # Filter samples to evaluate
    samples_to_eval = [
        data_id for data_id in common_data_ids
        if not skip_existing or data_id not in existing_data_ids
    ]

    if len(samples_to_eval) == 0:
        print(f"  All samples already evaluated for {model_a} vs {model_b}!")
        return {"total": len(common_data_ids), "success": len(existing_data_ids), "failed": 0}

    print(f"  Evaluating {len(samples_to_eval)} comparisons with {num_workers} workers...")

    # Open output file in append mode
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(samples_to_eval), "success": 0, "failed": 0}

    # Process samples in parallel and write results immediately
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_data_id = {
            executor.submit(
                compare_single_sample,
                data_id,
                model_a,
                model_b,
                metadata_by_model,
                base_dirs,
                evaluator,
                verbose=False
            ): data_id
            for data_id in samples_to_eval
        }

        with tqdm(total=len(samples_to_eval), desc=f"  {model_a} vs {model_b}") as pbar:
            with open(output_file, 'a', encoding='utf-8') as f:
                for future in as_completed(future_to_data_id):
                    result = future.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": stats["success"], "failed": stats["failed"]})

    return stats


def generate_pairwise_combinations(model_names: List[str]) -> List[Tuple[str, str]]:
    """Generate all pairwise combinations of models."""
    return list(combinations(model_names, 2))


def parse_model_pairs(model_pairs_str: str, available_models: List[str]) -> List[Tuple[str, str]]:
    """
    Parse model_pairs argument.

    Args:
        model_pairs_str: "all" or "model1,model2;model3,model4"
        available_models: List of available model names

    Returns:
        List of (model_a, model_b) tuples
    """
    if model_pairs_str.lower() == "all":
        return generate_pairwise_combinations(available_models)

    pairs = []
    for pair_str in model_pairs_str.split(";"):
        models = [m.strip() for m in pair_str.split(",")]
        if len(models) != 2:
            print(f"Warning: Invalid pair format '{pair_str}', expected 'model1,model2'")
            continue
        if models[0] not in available_models or models[1] not in available_models:
            print(f"Warning: Unknown model in pair '{pair_str}'")
            continue
        pairs.append((models[0], models[1]))

    return pairs


def discover_model_directories(input_dir: Path) -> List[str]:
    """Discover all model directories containing metadata.jsonl."""
    model_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and (item / "metadata.jsonl").exists():
            model_dirs.append(item.name)
    return sorted(model_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="DenseAlignBench — Pairwise Preference Evaluation for Image Generation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Base directory containing model subdirectories (each with metadata.jsonl + images)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save pairwise comparison results"
    )

    parser.add_argument(
        "--model_pairs",
        type=str,
        default="all",
        help='Model pairs to compare: "all" for all combinations, or "model1,model2;model3,model4" for specific pairs'
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the VLM endpoint (default: reads from GEMINI_API_KEY or OPENAI_API_KEY env var)"
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="OpenAI-compatible API endpoint (default: Google Gemini)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name for the evaluator (default: gemini-2.0-flash)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for API calls (default: 0.3)"
    )

    parser.add_argument(
        "--request_delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed API calls (default: 3)"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples already evaluated (resume from previous run)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing, default: all samples)"
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("DenseAlignBench — Pairwise Preference Evaluation")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.num_workers}")
    print(f"Temperature: {args.temperature}")
    print(f"Request delay: {args.request_delay}s")
    print(f"Max retries: {args.max_retries}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 80)

    # Discover model directories
    print("\nDiscovering model directories...")
    available_models = discover_model_directories(input_dir)
    if not available_models:
        print(f"Error: No model directories with metadata.jsonl found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(available_models)} models:")
    for model in available_models:
        print(f"  - {model}")

    # Parse model pairs
    print("\nParsing model pairs...")
    model_pairs = parse_model_pairs(args.model_pairs, available_models)
    if not model_pairs:
        print("Error: No valid model pairs to evaluate")
        sys.exit(1)

    print(f"Will evaluate {len(model_pairs)} model pairs:")
    for model_a, model_b in model_pairs:
        print(f"  - {model_a} vs {model_b}")

    # Load metadata from all models
    print("\nLoading metadata...")
    models_to_load = set()
    for model_a, model_b in model_pairs:
        models_to_load.add(model_a)
        models_to_load.add(model_b)

    metadata_by_model = load_model_metadata(input_dir, sorted(list(models_to_load)))

    if not metadata_by_model:
        print("Error: Failed to load metadata from any model")
        sys.exit(1)

    # Find common samples
    print("\nFinding common samples...")
    common_data_ids = find_common_samples(metadata_by_model)
    print(f"Found {len(common_data_ids)} common samples across all models")

    if len(common_data_ids) == 0:
        print("Error: No common samples found across models")
        sys.exit(1)

    # Limit samples if max_samples specified (for testing)
    if args.max_samples is not None and args.max_samples < len(common_data_ids):
        print(f"Limiting to first {args.max_samples} samples for testing")
        common_data_ids = common_data_ids[:args.max_samples]

    # Create base_dirs dict
    base_dirs = {model: input_dir / model for model in metadata_by_model.keys()}

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = DenseAlignEvaluator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        request_delay=args.request_delay,
        max_retries=args.max_retries
    )

    # Process each model pair
    print("\n" + "=" * 80)
    print("Starting Pairwise Evaluations")
    print("=" * 80)

    overall_stats = {
        "total_pairs": len(model_pairs),
        "total_comparisons": len(model_pairs) * len(common_data_ids),
        "success": 0,
        "failed": 0
    }

    start_time = time.time()

    for model_a, model_b in model_pairs:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_a} vs {model_b}")
        print(f"{'='*80}")

        output_file = output_dir / f"pairwise_comparison_{model_a}_vs_{model_b}.jsonl"

        pair_start_time = time.time()
        stats = evaluate_model_pair(
            model_a=model_a,
            model_b=model_b,
            common_data_ids=common_data_ids,
            metadata_by_model=metadata_by_model,
            base_dirs=base_dirs,
            evaluator=evaluator,
            output_file=output_file,
            num_workers=args.num_workers,
            skip_existing=args.skip_existing
        )
        pair_elapsed = time.time() - pair_start_time

        overall_stats["success"] += stats["success"]
        overall_stats["failed"] += stats["failed"]

        print(f"\n  Pair Summary:")
        print(f"    Total: {stats['total']}")
        print(f"    Success: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)")
        print(f"    Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.2f}%)")
        print(f"    Time: {pair_elapsed:.1f}s ({pair_elapsed/60:.1f}m)")
        print(f"    Speed: {stats['total']/pair_elapsed:.2f} comparisons/sec")
        print(f"    Output: {output_file}")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"Total model pairs: {overall_stats['total_pairs']}")
    print(f"Total comparisons: {overall_stats['total_comparisons']}")
    print(f"Success: {overall_stats['success']} ({overall_stats['success']/overall_stats['total_comparisons']*100:.2f}%)")
    print(f"Failed: {overall_stats['failed']} ({overall_stats['failed']/overall_stats['total_comparisons']*100:.2f}%)")
    print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    print(f"Average speed: {overall_stats['total_comparisons']/elapsed_time:.2f} comparisons/sec")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
