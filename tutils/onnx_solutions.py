#!/usr/bin/env python3
"""
ONNX Runtime Session Initialization - Solutions and Workarounds
================================================================

This file contains proven solutions for:
1. Slow session initialization (1+ hour for large models)
2. Linear scaling when initializing in parallel threads
3. GIL and internal locking issues

Based on research from ONNX Runtime GitHub issues and documentation.
"""

import onnxruntime as ort
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional
import pickle
import os


# =============================================================================
# SOLUTION 1: Optimize CUDA Provider Options
# =============================================================================
# The most common cause of slow initialization is cudnn_conv_algo_search
# The default 'EXHAUSTIVE' setting can take VERY long for models with convolutions

def create_fast_cuda_session(model_path: str, gpu_id: int = 0):
    """
    Create session with optimized CUDA options for faster initialization.
    
    Key optimization: cudnn_conv_algo_search
    - 'EXHAUSTIVE' (default): Tests ALL algorithms. Very slow but optimal inference.
    - 'HEURISTIC': Uses heuristics. Much faster init, good inference speed.
    - 'DEFAULT': Uses cuDNN defaults. Fastest init.
    """
    sess_opts = ort.SessionOptions()
    
    # Reduce logging during init
    sess_opts.log_severity_level = 3  # Error only
    
    cuda_options = {
        'device_id': gpu_id,
        
        # THIS IS THE KEY OPTIMIZATION for slow init
        # Change from 'EXHAUSTIVE' to 'HEURISTIC' or 'DEFAULT'
        'cudnn_conv_algo_search': 'HEURISTIC',
        
        # Other useful options:
        'arena_extend_strategy': 'kSameAsRequested',  # Can reduce memory fragmentation
        'cudnn_conv_use_max_workspace': '0',  # Reduce memory, may speed up init
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit (adjust as needed)
    }
    
    providers = [
        ('CUDAExecutionProvider', cuda_options),
        'CPUExecutionProvider'
    ]
    
    return ort.InferenceSession(model_path, sess_opts, providers=providers)


# =============================================================================
# SOLUTION 2: Save and Reuse Optimized Models
# =============================================================================
# Graph optimization during init is expensive. Do it once, save, reuse.

def create_and_save_optimized_model(
    input_model_path: str, 
    output_model_path: str,
    gpu_id: int = 0
):
    """
    Create an optimized model file that loads much faster.
    
    The first load does all the optimization work.
    Subsequent loads from the optimized file are much faster.
    """
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.optimized_model_filepath = output_model_path
    
    providers = [
        ('CUDAExecutionProvider', {'device_id': gpu_id}),
        'CPUExecutionProvider'
    ]
    
    # This creates the session AND saves the optimized model
    print(f"Creating optimized model (this may take a while)...")
    start = time.perf_counter()
    sess = ort.InferenceSession(input_model_path, sess_opts, providers=providers)
    print(f"Done in {time.perf_counter() - start:.2f}s")
    print(f"Optimized model saved to: {output_model_path}")
    
    return sess


def load_optimized_model(optimized_model_path: str, gpu_id: int = 0):
    """
    Load a pre-optimized model - should be much faster.
    """
    sess_opts = ort.SessionOptions()
    # Disable optimization since model is already optimized
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': gpu_id,
            'cudnn_conv_algo_search': 'HEURISTIC',
        }),
        'CPUExecutionProvider'
    ]
    
    return ort.InferenceSession(optimized_model_path, sess_opts, providers=providers)


# =============================================================================
# SOLUTION 3: Multiprocess Initialization (Bypasses GIL)
# =============================================================================
# Threading doesn't help because of GIL/internal locking.
# Use multiprocessing with spawn context instead.

def _init_session_worker(args):
    """Worker function for multiprocess session initialization."""
    model_path, gpu_id, cuda_options = args
    
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    
    default_cuda_opts = {
        'device_id': gpu_id,
        'cudnn_conv_algo_search': 'HEURISTIC',
    }
    default_cuda_opts.update(cuda_options or {})
    
    providers = [
        ('CUDAExecutionProvider', default_cuda_opts),
        'CPUExecutionProvider'
    ]
    
    start = time.perf_counter()
    sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
    init_time = time.perf_counter() - start
    
    # Return some session info (can't return the session itself)
    return {
        'gpu_id': gpu_id,
        'init_time': init_time,
        'inputs': [(i.name, i.shape, i.type) for i in sess.get_inputs()],
        'outputs': [(o.name, o.shape, o.type) for o in sess.get_outputs()],
    }


class MultiprocessSessionManager:
    """
    Manages ONNX sessions across multiple GPUs using separate processes.
    
    This is the recommended approach when you need to:
    - Initialize multiple sessions in parallel
    - Each session runs on a different GPU
    - You need to share large batches of data across sessions
    """
    
    def __init__(self, model_path: str, gpu_ids: List[int], cuda_options: dict = None):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.cuda_options = cuda_options or {}
        self.processes = {}
        self.request_queues = {}
        self.response_queues = {}
        
    def start(self):
        """Start worker processes for each GPU."""
        ctx = multiprocessing.get_context('spawn')
        
        for gpu_id in self.gpu_ids:
            req_queue = ctx.Queue()
            resp_queue = ctx.Queue()
            
            proc = ctx.Process(
                target=self._worker_loop,
                args=(self.model_path, gpu_id, self.cuda_options, req_queue, resp_queue)
            )
            proc.start()
            
            self.processes[gpu_id] = proc
            self.request_queues[gpu_id] = req_queue
            self.response_queues[gpu_id] = resp_queue
        
        # Wait for workers to be ready
        for gpu_id in self.gpu_ids:
            status = self.response_queues[gpu_id].get()
            print(f"GPU {gpu_id} worker ready: {status}")
    
    @staticmethod
    def _worker_loop(model_path, gpu_id, cuda_options, req_queue, resp_queue):
        """Worker process main loop."""
        import onnxruntime as ort
        import numpy as np
        
        # Initialize session
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        
        default_cuda_opts = {
            'device_id': gpu_id,
            'cudnn_conv_algo_search': 'HEURISTIC',
        }
        default_cuda_opts.update(cuda_options)
        
        providers = [
            ('CUDAExecutionProvider', default_cuda_opts),
            'CPUExecutionProvider'
        ]
        
        start = time.perf_counter()
        sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
        init_time = time.perf_counter() - start
        
        # Signal ready
        resp_queue.put({'status': 'ready', 'init_time': init_time})
        
        # Process requests
        while True:
            request = req_queue.get()
            
            if request['type'] == 'shutdown':
                break
            elif request['type'] == 'run':
                try:
                    inputs = request['inputs']
                    outputs = sess.run(None, inputs)
                    resp_queue.put({'status': 'success', 'outputs': outputs})
                except Exception as e:
                    resp_queue.put({'status': 'error', 'error': str(e)})
    
    def run(self, gpu_id: int, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference on specified GPU."""
        self.request_queues[gpu_id].put({'type': 'run', 'inputs': inputs})
        response = self.response_queues[gpu_id].get()
        
        if response['status'] == 'error':
            raise RuntimeError(f"Inference failed: {response['error']}")
        
        return response['outputs']
    
    def shutdown(self):
        """Shutdown all worker processes."""
        for gpu_id in self.gpu_ids:
            self.request_queues[gpu_id].put({'type': 'shutdown'})
        
        for proc in self.processes.values():
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


# =============================================================================
# SOLUTION 4: Shared Memory for Large Batch Data
# =============================================================================
# When using multiprocessing, sharing large arrays through queues is slow.
# Use shared memory instead.

class SharedMemoryBatchManager:
    """
    Efficiently share large batch data between processes using shared memory.
    """
    
    def __init__(self, batch_shape: tuple, dtype=np.float32):
        self.batch_shape = batch_shape
        self.dtype = dtype
        self.nbytes = int(np.prod(batch_shape) * np.dtype(dtype).itemsize)
        self.shm = None
        
    def create(self, name: str = None):
        """Create shared memory block."""
        self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes, name=name)
        return self.shm.name
    
    def attach(self, name: str):
        """Attach to existing shared memory block."""
        self.shm = shared_memory.SharedMemory(name=name)
    
    def get_array(self) -> np.ndarray:
        """Get numpy array backed by shared memory."""
        return np.ndarray(self.batch_shape, dtype=self.dtype, buffer=self.shm.buf)
    
    def close(self):
        """Close shared memory."""
        if self.shm:
            self.shm.close()
    
    def unlink(self):
        """Unlink (delete) shared memory."""
        if self.shm:
            self.shm.unlink()


# =============================================================================
# SOLUTION 5: IO Binding for Faster GPU Inference
# =============================================================================
# Once session is created, use IO binding to avoid CPU-GPU data transfers.

def run_with_io_binding(sess, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Run inference using IO binding for better GPU performance.
    
    This avoids unnecessary CPU-GPU memory copies.
    """
    io_binding = sess.io_binding()
    
    # Bind inputs
    for name, arr in inputs.items():
        # Create OrtValue on GPU
        ort_value = ort.OrtValue.ortvalue_from_numpy(arr, 'cuda', 0)
        io_binding.bind_ortvalue_input(name, ort_value)
    
    # Bind outputs (let ORT allocate on GPU)
    for output in sess.get_outputs():
        io_binding.bind_output(output.name, 'cuda', 0)
    
    # Run
    sess.run_with_iobinding(io_binding)
    
    # Get outputs
    return io_binding.copy_outputs_to_cpu()


# =============================================================================
# SOLUTION 6: Disable Graph Optimization for Faster Init (Trade-off)
# =============================================================================

def create_fast_init_session(model_path: str, gpu_id: int = 0):
    """
    Create session with minimal initialization time.
    
    Trade-off: Inference may be slower, but initialization is much faster.
    Good for development/testing or when you need quick startup.
    """
    sess_opts = ort.SessionOptions()
    
    # Disable all optimizations - fastest init but slowest inference
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    # Minimal threading
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': gpu_id,
            'cudnn_conv_algo_search': 'DEFAULT',  # Fastest option
        }),
        'CPUExecutionProvider'
    ]
    
    return ort.InferenceSession(model_path, sess_opts, providers=providers)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_parallel_init_with_multiprocessing():
    """Example: Initialize sessions on multiple GPUs in parallel using multiprocessing."""
    
    model_path = "model.onnx"
    gpu_ids = [0, 1, 2, 3]  # Adjust to your setup
    
    # Use ProcessPoolExecutor with spawn context
    ctx = multiprocessing.get_context('spawn')
    
    print(f"Initializing sessions on {len(gpu_ids)} GPUs in parallel...")
    start = time.perf_counter()
    
    args = [(model_path, gpu_id, {'cudnn_conv_algo_search': 'HEURISTIC'}) 
            for gpu_id in gpu_ids]
    
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as executor:
        results = list(executor.map(_init_session_worker, args))
    
    total_time = time.perf_counter() - start
    
    print(f"\nResults:")
    for r in results:
        print(f"  GPU {r['gpu_id']}: {r['init_time']:.2f}s")
    print(f"\nTotal wall time: {total_time:.2f}s")
    print(f"Sum of init times: {sum(r['init_time'] for r in results):.2f}s")


def example_optimized_workflow():
    """Example: Complete optimized workflow for production."""
    
    model_path = "model.onnx"
    optimized_path = "model_optimized.onnx"
    
    # Step 1: Create optimized model (do this once, offline)
    if not os.path.exists(optimized_path):
        print("Creating optimized model (one-time operation)...")
        create_and_save_optimized_model(model_path, optimized_path)
    
    # Step 2: Load optimized model with fast CUDA options
    print("\nLoading optimized model...")
    start = time.perf_counter()
    sess = load_optimized_model(optimized_path)
    print(f"Loaded in {time.perf_counter() - start:.2f}s")
    
    # Step 3: Run inference
    # ... your inference code here ...


if __name__ == '__main__':
    print("This file contains solutions and examples.")
    print("Import the functions you need or run the examples.")
    print("\nAvailable solutions:")
    print("  1. create_fast_cuda_session() - Optimized CUDA options")
    print("  2. create_and_save_optimized_model() - Pre-optimize models")
    print("  3. MultiprocessSessionManager - Parallel init across GPUs")
    print("  4. SharedMemoryBatchManager - Efficient data sharing")
    print("  5. run_with_io_binding() - Fast GPU inference")
    print("  6. create_fast_init_session() - Minimal init time")
