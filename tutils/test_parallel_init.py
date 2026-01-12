#!/usr/bin/env python3
"""Test threading vs multiprocessing for ONNX session initialization."""

import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import onnxruntime as ort

MODEL_PATH = '/workspace/models/resnet50_optimized.onnx'
NUM_WORKERS = 4


def create_session_cpu(worker_id):
    """Create a session in a worker."""
    start = time.perf_counter()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=['CPUExecutionProvider'])
    duration = time.perf_counter() - start
    del sess
    return {'worker_id': worker_id, 'duration': duration}


def main():
    print('=== Solution B: Threading vs Multiprocessing Comparison ===')
    print(f'Model: {MODEL_PATH}')
    print(f'Workers: {NUM_WORKERS}')
    print()

    # Test 1: Sequential
    print(f'Test 1: Sequential initialization...')
    start = time.perf_counter()
    seq_results = [create_session_cpu(i) for i in range(NUM_WORKERS)]
    seq_time = time.perf_counter() - start
    print(f'  Total time: {seq_time:.3f}s')
    durations = [f'{r["duration"]:.3f}s' for r in seq_results]
    print(f'  Individual: {durations}')

    # Test 2: Threading
    print('\nTest 2: Threaded initialization...')
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        thread_results = list(executor.map(create_session_cpu, range(NUM_WORKERS)))
    thread_time = time.perf_counter() - start
    print(f'  Total time: {thread_time:.3f}s')
    durations = [f'{r["duration"]:.3f}s' for r in thread_results]
    print(f'  Individual: {durations}')

    # Test 3: Multiprocessing
    print('\nTest 3: Multiprocess initialization...')
    ctx = multiprocessing.get_context('spawn')
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        mp_results = list(executor.map(create_session_cpu, range(NUM_WORKERS)))
    mp_time = time.perf_counter() - start
    print(f'  Total time: {mp_time:.3f}s')
    durations = [f'{r["duration"]:.3f}s' for r in mp_results]
    print(f'  Individual: {durations}')

    print()
    print('=== Results Summary ===')
    print(f'  Sequential:      {seq_time:.3f}s (baseline)')
    print(f'  Threading:       {thread_time:.3f}s ({seq_time/thread_time:.2f}x speedup)')
    print(f'  Multiprocessing: {mp_time:.3f}s ({seq_time/mp_time:.2f}x speedup)')
    print()

    thread_efficiency = (seq_time / thread_time - 1) / (NUM_WORKERS - 1) * 100
    mp_efficiency = (seq_time / mp_time - 1) / (NUM_WORKERS - 1) * 100

    print(f'  Threading parallelism efficiency:      {thread_efficiency:.1f}%')
    print(f'  Multiprocessing parallelism efficiency: {mp_efficiency:.1f}%')
    print()

    if mp_time < thread_time * 0.9:
        print('  ✓ Multiprocessing is faster than threading!')
        print('  → This confirms GIL/locking is the bottleneck')
        print('  → Use ProcessPoolExecutor with spawn context for parallel init')
    else:
        print('  Threading and multiprocessing have similar performance')
        print('  → GIL may not be the main bottleneck for this workload')


if __name__ == '__main__':
    main()
