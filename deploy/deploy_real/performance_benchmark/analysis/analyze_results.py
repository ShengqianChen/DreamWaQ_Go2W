#!/usr/bin/env python3
"""
GO2W Python vs C++ benchmark comparison script.
"""

import json
import argparse
import os

def load_results(filepath):
    """Load a benchmark JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_results(python_data, cpp_data):
    """Compare performance metrics between Python and C++."""
    py_timing = python_data.get("timing", {})
    cpp_timing = cpp_data.get("timing", {})

    results = {
        "timing_comparison": {},
        "control_quality_comparison": {},
        "summary": {}
    }


    metrics = [
        ("loop_time_ms", "Loop time"),
        ("inference_time_ms", "Inference time"),
        ("encoder_time_ms", "Encoder time"),
        ("actor_time_ms", "Actor time"),
    ]

    for key, name in metrics:
        py_val = py_timing.get(key, {}).get("mean", 0)
        cpp_val = cpp_timing.get(key, {}).get("mean", 0)

        if py_val > 0 and cpp_val > 0:
            speedup = py_val / cpp_val
            reduction = (py_val - cpp_val) / py_val * 100
        else:
            speedup = 0
            reduction = 0

        results["timing_comparison"][key] = {
            "name": name,
            "python_ms": py_val,
            "cpp_ms": cpp_val,
            "speedup": speedup,
            "reduction_percent": reduction
        }


    py_cq = python_data.get("control_quality", {})
    cpp_cq = cpp_data.get("control_quality", {})

    results["control_quality_comparison"] = {
        "python_mean_error": py_cq.get("mean_error", 0),
        "cpp_mean_error": cpp_cq.get("mean_error", 0),
        "python_max_error": py_cq.get("max_error", 0),
        "cpp_max_error": cpp_cq.get("max_error", 0),
    }


    py_samples = python_data.get("metadata", {}).get("total_samples", 0)
    cpp_samples = cpp_data.get("metadata", {}).get("total_samples", 0)


    loop_speedup = results["timing_comparison"]["loop_time_ms"]["speedup"]
    inference_speedup = results["timing_comparison"]["inference_time_ms"]["speedup"]

    results["summary"] = {
        "python_samples": py_samples,
        "cpp_samples": cpp_samples,
        "loop_speedup": loop_speedup,
        "inference_speedup": inference_speedup,
        "cpp_is_faster": loop_speedup > 1.0
    }

    return results

def print_report(comparison):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("        GO2W Python vs C++ Performance Report")
    print("=" * 70)

    summary = comparison["summary"]
    print(f"\nSamples: Python={summary['python_samples']}, C++={summary['cpp_samples']}")


    print("\n" + "-" * 70)
    print("Timing Comparison")
    print("-" * 70)
    print(f"{'Metric':<20} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<12} {'Reduction %':<10}")
    print("-" * 70)

    for key, data in comparison["timing_comparison"].items():
        name = data["name"]
        py = data["python_ms"]
        cpp = data["cpp_ms"]
        speedup = data["speedup"]
        reduction = data["reduction_percent"]

        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "-"
        reduction_str = f"{reduction:.1f}%" if reduction != 0 else "-"

        print(f"{name:<20} {py:<15.3f} {cpp:<15.3f} {speedup_str:<12} {reduction_str:<10}")


    cq = comparison["control_quality_comparison"]
    print("\n" + "-" * 70)
    print("Control Quality Comparison")
    print("-" * 70)
    print(f"{'Metric':<25} {'Python':<20} {'C++':<20}")
    print("-" * 70)
    print(f"{'Mean tracking error (rad)':<25} {cq['python_mean_error']:<20.6f} {cq['cpp_mean_error']:<20.6f}")
    print(f"{'Max tracking error (rad)':<25} {cq['python_max_error']:<20.6f} {cq['cpp_max_error']:<20.6f}")


    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)

    if summary["cpp_is_faster"]:
        print(f"✓ C++ loop execution is faster ({summary['loop_speedup']:.2f}x)")
        print(f"✓ C++ inference is faster ({summary['inference_speedup']:.2f}x)")
    else:
        print(f"✗ Python loop execution is faster ({1/summary['loop_speedup']:.2f}x)")

    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='GO2W benchmark comparison')
    parser.add_argument('python_result', type=str, help='Path to Python benchmark JSON')
    parser.add_argument('cpp_result', type=str, help='Path to C++ benchmark JSON')
    parser.add_argument('--output', type=str, help='Optional output JSON path for comparison results')
    args = parser.parse_args()


    python_data = load_results(args.python_result)
    cpp_data = load_results(args.cpp_result)


    comparison = compare_results(python_data, cpp_data)


    print_report(comparison)


    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nComparison results saved to: {args.output}")

if __name__ == "__main__":
    main()
