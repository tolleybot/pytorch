#!/usr/bin/env python3
"""
Quick GIL Detection Test for ONNX Runtime
==========================================

This script quickly tests whether GIL holding is the cause of
linear scaling in threaded ONNX session initialization.

Run with: python test_gil_detection.py

This test does NOT require a model file - it simulates the behavior.
"""

import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


def cpu_bound_work(duration_ms: int, task_id: int) -> dict:
    """Simulates CPU-bound work (like model loading)."""
    start = time.perf_counter()
    
    # Busy loop to simulate CPU work
    # This is intentionally inefficient to demonstrate GIL behavior
    count = 0
    target = duration_ms * 10000  # Rough calibration
    while count < target:
        count += 1
    
    end = time.perf_counter()
    return {
        'task_id': task_id,
        'duration': end - start,
        'thread': threading.current_thread().name,
        'start': start,
        'end': end
    }


def io_bound_work(duration_ms: int, task_id: int) -> dict:
    """Simulates I/O-bound work."""
    start = time.perf_counter()
    time.sleep(duration_ms / 1000)
    end = time.perf_counter()
    return {
        'task_id': task_id,
        'duration': end - start,
        'thread': threading.current_thread().name,
        'start': start,
        'end': end
    }


def test_threading_with_pure_python():
    """
    Test 1: Pure Python CPU-bound work with threading.
    
    Expected: Linear scaling (GIL prevents parallelism)
    """
    print("\n[TEST 1] Pure Python CPU-bound work with threading")
    print("-" * 50)
    print("(This should show LINEAR scaling due to GIL)")
    
    n_tasks = 4
    work_ms = 200
    
    # Sequential baseline
    start = time.perf_counter()
    for i in range(n_tasks):
        cpu_bound_work(work_ms, i)
    seq_time = time.perf_counter() - start
    print(f"  Sequential: {seq_time:.3f}s")
    
    # Threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as ex:
        list(ex.map(lambda i: cpu_bound_work(work_ms, i), range(n_tasks)))
    thread_time = time.perf_counter() - start
    print(f"  Threaded:   {thread_time:.3f}s")
    
    speedup = seq_time / thread_time
    print(f"  Speedup:    {speedup:.2f}x (ideal: {n_tasks}x)")
    
    if speedup < 1.5:
        print("  → GIL is blocking parallelism (as expected for pure Python)")
    return speedup


def test_threading_with_io():
    """
    Test 2: I/O-bound work with threading.
    
    Expected: Good parallelism (GIL released during I/O)
    """
    print("\n[TEST 2] I/O-bound work with threading")
    print("-" * 50)
    print("(This should show GOOD parallelism - GIL released during I/O)")
    
    n_tasks = 4
    work_ms = 200
    
    # Sequential baseline
    start = time.perf_counter()
    for i in range(n_tasks):
        io_bound_work(work_ms, i)
    seq_time = time.perf_counter() - start
    print(f"  Sequential: {seq_time:.3f}s")
    
    # Threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as ex:
        list(ex.map(lambda i: io_bound_work(work_ms, i), range(n_tasks)))
    thread_time = time.perf_counter() - start
    print(f"  Threaded:   {thread_time:.3f}s")
    
    speedup = seq_time / thread_time
    print(f"  Speedup:    {speedup:.2f}x (ideal: {n_tasks}x)")
    
    if speedup > 2:
        print("  → Good parallelism (GIL released during I/O)")
    return speedup


def _process_worker(args):
    """Worker for multiprocessing test."""
    work_ms, task_id = args
    return cpu_bound_work(work_ms, task_id)


def test_multiprocessing():
    """
    Test 3: CPU-bound work with multiprocessing.
    
    Expected: Good parallelism (separate processes, separate GILs)
    """
    print("\n[TEST 3] CPU-bound work with multiprocessing")
    print("-" * 50)
    print("(This should show GOOD parallelism - separate GILs)")
    
    n_tasks = 4
    work_ms = 200
    
    # Sequential baseline (already measured above, use approximation)
    start = time.perf_counter()
    for i in range(n_tasks):
        cpu_bound_work(work_ms, i)
    seq_time = time.perf_counter() - start
    print(f"  Sequential: {seq_time:.3f}s")
    
    # Multiprocess
    ctx = multiprocessing.get_context('spawn')
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_tasks, mp_context=ctx) as ex:
        list(ex.map(_process_worker, [(work_ms, i) for i in range(n_tasks)]))
    mp_time = time.perf_counter() - start
    print(f"  Multiproc:  {mp_time:.3f}s")
    
    speedup = seq_time / mp_time
    print(f"  Speedup:    {speedup:.2f}x (ideal: {n_tasks}x)")
    
    if speedup > 2:
        print("  → Good parallelism (multiprocessing bypasses GIL)")
    return speedup


def test_onnx_session_init_if_available():
    """
    Test 4: Actual ONNX Runtime session initialization.
    
    This will show if ONNX Runtime holds the GIL during init.
    """
    print("\n[TEST 4] ONNX Runtime session initialization (if available)")
    print("-" * 50)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ONNX Runtime not installed - skipping")
        return None
    
    # Check for CUDA
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    
    if 'CUDAExecutionProvider' not in providers:
        print("  CUDA not available - testing with CPU only")
    
    # We need a model to test. Try to create a simple one.
    try:
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
        import tempfile
        
        # Create minimal model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])
        
        W = helper.make_tensor('W', TensorProto.FLOAT, [64, 64],
                               np.random.randn(64, 64).astype(np.float32).flatten().tolist())
        
        matmul = helper.make_node('MatMul', ['X', 'W'], ['Y'])
        graph = helper.make_graph([matmul], 'test', [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
        model.ir_version = 8  # Set compatible IR version

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model, f.name)
            model_path = f.name
        
        print(f"  Created test model: {model_path}")
        
    except ImportError:
        print("  'onnx' package not installed - cannot create test model")
        print("  Please provide a model file using the main diagnostic script")
        return None
    
    n_sessions = 4
    
    def create_session(i):
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 4  # Fatal only
        sess_opts.intra_op_num_threads = 1
        start = time.perf_counter()
        sess = ort.InferenceSession(model_path, sess_opts, 
                                     providers=['CPUExecutionProvider'])
        end = time.perf_counter()
        del sess
        return {'id': i, 'duration': end - start, 'start': start, 'end': end}
    
    # Sequential
    print("\n  Testing session creation...")
    start = time.perf_counter()
    seq_results = [create_session(i) for i in range(n_sessions)]
    seq_time = time.perf_counter() - start
    print(f"  Sequential: {seq_time:.4f}s (avg: {seq_time/n_sessions:.4f}s per session)")
    
    # Threaded
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_sessions) as ex:
        thread_results = list(ex.map(create_session, range(n_sessions)))
    thread_time = time.perf_counter() - start
    print(f"  Threaded:   {thread_time:.4f}s")
    
    speedup = seq_time / thread_time
    print(f"  Speedup:    {speedup:.2f}x (ideal: {n_sessions}x)")
    
    # Analyze overlap
    min_start = min(r['start'] for r in thread_results)
    max_end = max(r['end'] for r in thread_results)
    
    # Check if they ran concurrently
    total_individual = sum(r['duration'] for r in thread_results)
    wall_time = max_end - min_start
    concurrency = total_individual / wall_time if wall_time > 0 else 1
    
    print(f"  Concurrency factor: {concurrency:.2f}")
    
    if concurrency > 1.5:
        print("  → Sessions are running concurrently (GIL released during native code)")
    else:
        print("  → Sessions are running serially (GIL likely held during init)")
    
    # Clean up
    import os
    os.unlink(model_path)
    
    return speedup


def main():
    print("="*70)
    print("GIL DETECTION TEST FOR ONNX RUNTIME")
    print("="*70)
    print("\nThis test helps determine if GIL holding is causing")
    print("linear scaling in threaded ONNX session initialization.\n")
    
    # Run tests
    py_speedup = test_threading_with_pure_python()
    io_speedup = test_threading_with_io()
    mp_speedup = test_multiprocessing()
    ort_speedup = test_onnx_session_init_if_available()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n  Pure Python CPU (threads):  {py_speedup:.2f}x speedup (expect ~1x)")
    print(f"  I/O-bound (threads):        {io_speedup:.2f}x speedup (expect ~4x)")
    print(f"  CPU-bound (multiprocess):   {mp_speedup:.2f}x speedup (expect ~4x)")
    
    if ort_speedup is not None:
        print(f"  ONNX Runtime init (threads): {ort_speedup:.2f}x speedup")
        
        if ort_speedup < 1.5:
            print("\n  DIAGNOSIS: ONNX Runtime session initialization likely holds the GIL")
            print("  SOLUTION: Use multiprocessing instead of threading")
        elif ort_speedup < 3:
            print("\n  DIAGNOSIS: Partial GIL holding or internal locking in ONNX Runtime")
            print("  SOLUTION: Use multiprocessing for better parallelism")
        else:
            print("\n  DIAGNOSIS: ONNX Runtime init shows good parallelism")
            print("  The issue may be CUDA-specific or model-specific")
    
    print("\n  For full diagnosis with your actual model, run:")
    print("    python onnx_session_diagnostics.py --model your_model.onnx")
    print("="*70)


if __name__ == '__main__':
    main()
