#!/usr/bin/env python3
"""Test threading vs multiprocessing with large Phi-3 model."""

import onnxruntime as ort
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

MODEL_PATH = '/workspace/models/phi3-mini-cuda-fp16/cuda/cuda-fp16/phi3-mini-4k-instruct-cuda-fp16.onnx'
NUM_GPUS = 4


def create_session(gpu_id):
    """Create a session on specified GPU."""
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = [
        ('CUDAExecutionProvider', {'device_id': gpu_id}),
        'CPUExecutionProvider'
    ]

    start = time.perf_counter()
    sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=providers)
    duration = time.perf_counter() - start
    del sess
    return {'gpu_id': gpu_id, 'duration': duration}


def main():
    print('=== Threading vs Multiprocessing with 7.2GB Phi-3 Model ===')
    print(f'Model: {MODEL_PATH}')
    print(f'Testing on {NUM_GPUS} GPUs')
    print()

    # Test 1: Sequential
    print('Test 1: Sequential initialization...')
    start = time.perf_counter()
    seq_results = [create_session(i) for i in range(NUM_GPUS)]
    seq_time = time.perf_counter() - start
    print(f'  Total time: {seq_time:.2f}s')
    for r in seq_results:
        print(f'    GPU {r["gpu_id"]}: {r["duration"]:.2f}s')

    print()

    # Test 2: Threading
    print('Test 2: Threaded initialization...')
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        thread_results = list(executor.map(create_session, range(NUM_GPUS)))
    thread_time = time.perf_counter() - start
    print(f'  Total wall time: {thread_time:.2f}s')
    for r in thread_results:
        print(f'    GPU {r["gpu_id"]}: {r["duration"]:.2f}s')

    print()

    # Test 3: Multiprocessing
    print('Test 3: Multiprocess initialization...')
    ctx = multiprocessing.get_context('spawn')
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_GPUS, mp_context=ctx) as executor:
        mp_results = list(executor.map(create_session, range(NUM_GPUS)))
    mp_time = time.perf_counter() - start
    print(f'  Total wall time: {mp_time:.2f}s')
    for r in mp_results:
        print(f'    GPU {r["gpu_id"]}: {r["duration"]:.2f}s')

    print()
    print('=== Summary ===')
    print(f'  Sequential:      {seq_time:.2f}s (baseline)')
    print(f'  Threaded:        {thread_time:.2f}s ({seq_time/thread_time:.2f}x speedup)')
    print(f'  Multiprocessing: {mp_time:.2f}s ({seq_time/mp_time:.2f}x speedup)')
    print()

    thread_eff = (seq_time / thread_time - 1) / (NUM_GPUS - 1) * 100
    mp_eff = (seq_time / mp_time - 1) / (NUM_GPUS - 1) * 100

    print(f'  Threading efficiency:      {thread_eff:.1f}%')
    print(f'  Multiprocessing efficiency: {mp_eff:.1f}%')

    if mp_time < thread_time * 0.9:
        print()
        print('  ✓ Multiprocessing is faster than threading!')
        print('  → This confirms GIL/locking affects large model initialization')


if __name__ == '__main__':
    main()
