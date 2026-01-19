#!/usr/bin/env python3
"""
Benchmark inference performance: HEURISTIC vs EXHAUSTIVE cudnn_conv_algo_search

This test measures whether using HEURISTIC (faster init) hurts inference throughput
compared to EXHAUSTIVE (slower init, optimal algorithm selection).
"""

import time
import numpy as np
import onnxruntime as ort

# Use ResNet50 for this test (conv-heavy model where algorithm choice matters most)
MODEL_PATH = "/workspace/models/resnet50-v2-7/resnet50-v2-7.onnx"

# Number of inference iterations for benchmarking
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def create_session(algo_search: str, use_optimized: bool = False):
    """Create session with specified cudnn_conv_algo_search setting."""
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3

    if use_optimized:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        model_path = "/workspace/models/resnet50_optimized.onnx"
    else:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model_path = MODEL_PATH

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'cudnn_conv_algo_search': algo_search,
        }),
        'CPUExecutionProvider'
    ]

    start = time.perf_counter()
    sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
    init_time = time.perf_counter() - start

    return sess, init_time


def benchmark_inference(sess, input_name, input_shape, num_iterations):
    """Run inference benchmark and return average latency."""
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        sess.run(None, {input_name: input_data})

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        sess.run(None, {input_name: input_data})
        latencies.append(time.perf_counter() - start)

    return latencies


def main():
    print("=" * 70)
    print("INFERENCE PERFORMANCE: HEURISTIC vs EXHAUSTIVE cudnn_conv_algo_search")
    print("=" * 70)
    print(f"\nModel: ResNet50-v2")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")
    print()

    # Check available providers
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print("ERROR: CUDAExecutionProvider not available")
        print(f"Available: {providers}")
        return

    results = {}

    for algo in ['EXHAUSTIVE', 'HEURISTIC', 'DEFAULT']:
        print(f"\n{'='*70}")
        print(f"Testing: cudnn_conv_algo_search = '{algo}'")
        print('='*70)

        # Create session
        print(f"Initializing session...")
        sess, init_time = create_session(algo)
        print(f"Init time: {init_time:.3f}s")

        # Get input info
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        input_shape = [1, 3, 224, 224]  # Standard ResNet input

        # Check provider
        actual_provider = sess.get_providers()[0]
        print(f"Provider: {actual_provider}")

        if actual_provider != 'CUDAExecutionProvider':
            print("WARNING: Not using CUDA, skipping...")
            continue

        # Run benchmark
        print(f"Running {BENCHMARK_ITERATIONS} inference iterations...")
        latencies = benchmark_inference(sess, input_name, input_shape, BENCHMARK_ITERATIONS)

        avg_latency = np.mean(latencies) * 1000  # ms
        std_latency = np.std(latencies) * 1000   # ms
        min_latency = np.min(latencies) * 1000   # ms
        max_latency = np.max(latencies) * 1000   # ms
        throughput = 1000 / avg_latency          # inferences/sec

        results[algo] = {
            'init_time': init_time,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'throughput': throughput,
        }

        print(f"\nResults for {algo}:")
        print(f"  Init time:       {init_time:.3f}s")
        print(f"  Avg latency:     {avg_latency:.3f}ms (±{std_latency:.3f}ms)")
        print(f"  Min/Max latency: {min_latency:.3f}ms / {max_latency:.3f}ms")
        print(f"  Throughput:      {throughput:.1f} inferences/sec")

        del sess

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    if 'EXHAUSTIVE' in results and 'HEURISTIC' in results:
        exh = results['EXHAUSTIVE']
        heu = results['HEURISTIC']

        init_speedup = exh['init_time'] / heu['init_time']
        latency_diff = ((heu['avg_latency'] - exh['avg_latency']) / exh['avg_latency']) * 100
        throughput_diff = ((heu['throughput'] - exh['throughput']) / exh['throughput']) * 100

        print(f"\n| Metric | EXHAUSTIVE | HEURISTIC | Difference |")
        print(f"|--------|------------|-----------|------------|")
        print(f"| Init time | {exh['init_time']:.3f}s | {heu['init_time']:.3f}s | {init_speedup:.2f}x faster |")
        print(f"| Avg latency | {exh['avg_latency']:.3f}ms | {heu['avg_latency']:.3f}ms | {latency_diff:+.1f}% |")
        print(f"| Throughput | {exh['throughput']:.1f}/s | {heu['throughput']:.1f}/s | {throughput_diff:+.1f}% |")

        print(f"\n**Key Finding:**")
        if abs(latency_diff) < 5:
            print(f"HEURISTIC provides {init_speedup:.1f}x faster init with <5% inference impact.")
            print("Recommendation: Use HEURISTIC for most production scenarios.")
        elif latency_diff > 0:
            print(f"HEURISTIC is {latency_diff:.1f}% slower at inference but {init_speedup:.1f}x faster to init.")
            print("Recommendation: Use EXHAUSTIVE if running many inferences, HEURISTIC for few.")
        else:
            print(f"HEURISTIC is actually {-latency_diff:.1f}% FASTER at inference!")
            print("Recommendation: Use HEURISTIC - faster init AND inference.")

        # Calculate break-even point
        if latency_diff > 0:
            init_savings_ms = (exh['init_time'] - heu['init_time']) * 1000
            latency_cost_ms = heu['avg_latency'] - exh['avg_latency']
            if latency_cost_ms > 0:
                breakeven = int(init_savings_ms / latency_cost_ms)
                print(f"\nBreak-even point: ~{breakeven:,} inferences")
                print(f"  - <{breakeven:,} inferences: HEURISTIC is better (init savings outweigh inference cost)")
                print(f"  - >{breakeven:,} inferences: EXHAUSTIVE is better (optimal algorithms pay off)")


if __name__ == "__main__":
    main()
