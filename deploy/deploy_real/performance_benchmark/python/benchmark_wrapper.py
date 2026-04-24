#!/usr/bin/env python3
"""
GO2W Python deployment benchmark wrapper (lightweight, no resource monitor).
"""

import time
import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class TimingStats:
    """Container for timing statistics."""
    samples: List[float] = field(default_factory=list)

    def add(self, value: float):
        self.samples.append(value)

    def get_stats(self) -> Dict:
        if not self.samples:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0, "percentile_99": 0}
        arr = np.array(self.samples)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
            "percentile_99": float(np.percentile(arr, 99)) if len(arr) > 10 else float(np.max(arr))
        }


@dataclass
class ControlQualityMetrics:
    """Control quality metrics."""
    target_positions: List[List[float]] = field(default_factory=list)
    actual_positions: List[List[float]] = field(default_factory=list)

    def add_sample(self, target: np.ndarray, actual: np.ndarray):
        self.target_positions.append(target.tolist())
        self.actual_positions.append(actual.tolist())

    def get_tracking_error(self) -> Dict:
        if not self.target_positions:
            return {"mean_error": 0, "max_error": 0}
        targets = np.array(self.target_positions)
        actuals = np.array(self.actual_positions)
        errors = np.abs(targets - actuals)
        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "per_joint_mean": np.mean(errors, axis=0).tolist()
        }


class PerformanceMonitor:
    """Lightweight performance monitor."""

    def __init__(self):
        self.loop_time = TimingStats()
        self.inference_time = TimingStats()
        self.encoder_time = TimingStats()
        self.actor_time = TimingStats()
        self.comm_send_time = TimingStats()
        self.control_quality = ControlQualityMetrics()

        self._loop_start = 0
        self._inference_start = 0
        self._encoder_start = 0
        self._actor_start = 0
        self._comm_start = 0


    def start_loop(self):
        self._loop_start = time.perf_counter()

    def end_loop(self):
        self.loop_time.add((time.perf_counter() - self._loop_start) * 1000)

    def start_inference(self):
        self._inference_start = time.perf_counter()

    def end_inference(self):
        self.inference_time.add((time.perf_counter() - self._inference_start) * 1000)

    def start_encoder(self):
        self._encoder_start = time.perf_counter()

    def end_encoder(self):
        self.encoder_time.add((time.perf_counter() - self._encoder_start) * 1000)

    def start_actor(self):
        self._actor_start = time.perf_counter()

    def end_actor(self):
        self.actor_time.add((time.perf_counter() - self._actor_start) * 1000)

    def start_comm_send(self):
        self._comm_start = time.perf_counter()

    def end_comm_send(self):
        self.comm_send_time.add((time.perf_counter() - self._comm_start) * 1000)

    def record_control_quality(self, target_pos: np.ndarray, actual_pos: np.ndarray):
        self.control_quality.add_sample(target_pos, actual_pos)


    def record_resource_usage(self):
        pass

    def get_results(self) -> Dict:
        """Return all benchmark results."""
        return {
            "timing": {
                "loop_time_ms": self.loop_time.get_stats(),
                "inference_time_ms": self.inference_time.get_stats(),
                "encoder_time_ms": self.encoder_time.get_stats(),
                "actor_time_ms": self.actor_time.get_stats(),
                "comm_send_time_ms": self.comm_send_time.get_stats(),
            },
            "control_quality": self.control_quality.get_tracking_error(),
            "metadata": {
                "language": "python",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": self.loop_time.get_stats()["count"]
            }
        }

    def save_results(self, filepath: str):
        """Save benchmark results to JSON."""
        results = self.get_results()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

    def print_summary(self):
        """Print a short benchmark summary."""
        results = self.get_results()
        print("\n" + "=" * 50)
        print("Python Performance Benchmark Results")
        print("=" * 50)

        timing = results["timing"]
        print(f"\n[Timing (ms)]")
        print(f"  Loop Time:      mean={timing['loop_time_ms']['mean']:.3f}, "
              f"max={timing['loop_time_ms']['max']:.3f}, "
              f"99th={timing['loop_time_ms'].get('percentile_99', 0):.3f}")
        print(f"  Inference:      mean={timing['inference_time_ms']['mean']:.3f}, "
              f"max={timing['inference_time_ms']['max']:.3f}")
        print(f"    - Encoder:    mean={timing['encoder_time_ms']['mean']:.3f}, "
              f"max={timing['encoder_time_ms']['max']:.3f}")
        print(f"    - Actor:      mean={timing['actor_time_ms']['mean']:.3f}, "
              f"max={timing['actor_time_ms']['max']:.3f}")

        cq = results["control_quality"]
        print(f"\n[Control Quality]")
        print(f"  Mean Tracking Error: {cq['mean_error']:.6f} rad")
        print(f"  Max Tracking Error:  {cq['max_error']:.6f} rad")

        print(f"\nTotal samples: {results['metadata']['total_samples']}")
        print("=" * 50)


_monitor: Optional[PerformanceMonitor] = None

def get_monitor() -> PerformanceMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


if __name__ == "__main__":

    monitor = PerformanceMonitor()
    print("Testing PerformanceMonitor...")

    for i in range(100):
        monitor.start_loop()
        time.sleep(0.01)

        monitor.start_inference()
        monitor.start_encoder()
        time.sleep(0.003)
        monitor.end_encoder()
        monitor.start_actor()
        time.sleep(0.002)
        monitor.end_actor()
        monitor.end_inference()

        monitor.end_loop()

        target = np.random.randn(12) * 0.1
        actual = target + np.random.randn(12) * 0.01
        monitor.record_control_quality(target, actual)

    monitor.print_summary()
    print("\nTest completed!")
