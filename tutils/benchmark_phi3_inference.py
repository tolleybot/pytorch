#!/usr/bin/env python3
"""
Benchmark inference performance on Phi-3 (7.2GB): HEURISTIC vs EXHAUSTIVE

Tests whether HEURISTIC impacts inference throughput on a large transformer model.
"""

import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/models/phi3-mini-cuda-fp16/cuda/cuda-fp16/phi3-mini-4k-instruct-cuda-fp16.onnx"

WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 20  # Fewer iterations since inference is slower


def create_session(algo_search: str):
    """Create session with specified cudnn_conv_algo_search setting."""
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'cudnn_conv_algo_search': algo_search,
        }),
        'CPUExecutionProvider'
    ]

    start = time.perf_counter()
    sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=providers)
    init_time = time.perf_counter() - start

    return sess, init_time


def benchmark_inference(sess, num_iterations):
    """Run inference benchmark and return average latency."""
    # Get input info
    inputs = sess.get_inputs()
    input_dict = {}

    print(f"  Model inputs:")
    for inp in inputs:
        print(f"    {inp.name}: {inp.shape} ({inp.type})")
        # Create appropriate input based on shape and type
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                # Dynamic dimension - use small value
                if 'batch' in str(dim).lower():
                    shape.append(1)
                elif 'sequence' in str(dim).lower() or 'seq' in str(dim).lower():
                    shape.append(32)  # Short sequence for benchmarking
                else:
                    shape.append(32)
            else:
                shape.append(dim)

        if 'int' in inp.type:
            input_dict[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            input_dict[inp.name] = np.random.randn(*shape).astype(np.float16)

    print(f"  Input shapes used: {[(k, v.shape) for k, v in input_dict.items()]}")

    # Warmup
    print(f"  Warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        sess.run(None, input_dict)

    # Benchmark
    print(f"  Benchmarking ({BENCHMARK_ITERATIONS} iterations)...")
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        sess.run(None, input_dict)
        latencies.append(time.perf_counter() - start)
        if (i + 1) % 5 == 0:
            print(f"    Completed {i + 1}/{num_iterations}")

    return latencies


def main():
    print("=" * 70)
    print("PHI-3 INFERENCE PERFORMANCE: HEURISTIC vs EXHAUSTIVE")
    print("=" * 70)
    print(f"\nModel: Phi-3-mini-4k-instruct (7.2GB)")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")

    # Check available providers
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print("\nERROR: CUDAExecutionProvider not available")
        print(f"Available: {providers}")
        return

    results = {}

    for algo in ['EXHAUSTIVE', 'HEURISTIC']:
        print(f"\n{'='*70}")
        print(f"Testing: cudnn_conv_algo_search = '{algo}'")
        print('='*70)

        # Create session
        print(f"\nInitializing session...")
        sess, init_time = create_session(algo)
        print(f"Init time: {init_time:.3f}s")

        # Check provider
        actual_provider = sess.get_providers()[0]
        print(f"Provider: {actual_provider}")

        if actual_provider != 'CUDAExecutionProvider':
            print("WARNING: Not using CUDA, skipping...")
            del sess
            continue

        # Run benchmark
        print(f"\nRunning inference benchmark...")
        latencies = benchmark_inference(sess, BENCHMARK_ITERATIONS)

        avg_latency = np.mean(latencies) * 1000  # ms
        std_latency = np.std(latencies) * 1000   # ms
        min_latency = np.min(latencies) * 1000   # ms
        max_latency = np.max(latencies) * 1000   # ms

        results[algo] = {
            'init_time': init_time,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
        }

        print(f"\nResults for {algo}:")
        print(f"  Init time:       {init_time:.3f}s")
        print(f"  Avg latency:     {avg_latency:.2f}ms (±{std_latency:.2f}ms)")
        print(f"  Min/Max latency: {min_latency:.2f}ms / {max_latency:.2f}ms")

        del sess

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON - Phi-3-mini (7.2GB)")
    print("=" * 70)

    if 'EXHAUSTIVE' in results and 'HEURISTIC' in results:
        exh = results['EXHAUSTIVE']
        heu = results['HEURISTIC']

        init_speedup = exh['init_time'] / heu['init_time']
        latency_diff_pct = ((heu['avg_latency'] - exh['avg_latency']) / exh['avg_latency']) * 100

        print(f"\n| Metric | EXHAUSTIVE | HEURISTIC | Difference |")
        print(f"|--------|------------|-----------|------------|")
        print(f"| Init time | {exh['init_time']:.2f}s | {heu['init_time']:.2f}s | {init_speedup:.1f}x faster |")
        print(f"| Avg latency | {exh['avg_latency']:.2f}ms | {heu['avg_latency']:.2f}ms | {latency_diff_pct:+.1f}% |")

        print(f"\n**Key Finding for Large Transformer Models:**")
        if abs(latency_diff_pct) < 5:
            print(f"HEURISTIC provides {init_speedup:.1f}x faster init with negligible inference impact (<5%).")
        elif latency_diff_pct > 0:
            print(f"HEURISTIC is {latency_diff_pct:.1f}% slower at inference but {init_speedup:.1f}x faster to init.")
        else:
            print(f"HEURISTIC is {-latency_diff_pct:.1f}% FASTER at inference AND {init_speedup:.1f}x faster to init.")

        # Note about transformer models
        print(f"\nNote: Phi-3 is a transformer model (attention-based, not conv-heavy).")
        print(f"cudnn_conv_algo_search primarily affects convolution layers.")
        print(f"For transformer models, the impact may be minimal compared to CNN models like ResNet.")


if __name__ == "__main__":
    main()
