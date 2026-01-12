#!/usr/bin/env python3
"""
ONNX Runtime Session Initialization Diagnostic Toolkit
=======================================================

This script diagnoses why ONNX Runtime session initialization:
1. Scales linearly when using threads (GIL or internal locking)
2. Takes a very long time for large models

Run with: python onnx_session_diagnostics.py --model path/to/your/model.onnx

Author: Diagnostic script for Casey's reported issue
"""

import argparse
import gc
import json
import os
import sys
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

# Check if onnxruntime is available
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("WARNING: onnxruntime not installed. Install with: pip install onnxruntime-gpu")

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


@dataclass
class TimingResult:
    """Store timing results for a single session initialization."""
    gpu_id: int
    duration_seconds: float
    thread_or_process_id: str
    start_time: float
    end_time: float
    success: bool
    error: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)


class DiagnosticRunner:
    """Run various diagnostics on ONNX Runtime session initialization."""
    
    def __init__(self, model_path: str, num_gpus: int = None):
        self.model_path = model_path
        self.num_gpus = num_gpus or self._detect_gpus()
        self.results = {}
        
    def _detect_gpus(self) -> int:
        """Detect number of available GPUs."""
        if not ORT_AVAILABLE:
            return 0
        
        # Try CUDA detection
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            pass
        
        # Fallback: try to create sessions on GPUs until it fails
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("CUDA Execution Provider not available")
            return 0
            
        count = 0
        for i in range(16):  # Max 16 GPUs
            try:
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 4  # Fatal only
                providers = [('CUDAExecutionProvider', {'device_id': i})]
                sess = ort.InferenceSession(self.model_path, sess_opts, providers=providers)
                del sess
                count += 1
            except Exception:
                break
        return max(count, 1)
    
    def print_system_info(self):
        """Print system and ONNX Runtime information."""
        print("\n" + "="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        
        if ORT_AVAILABLE:
            print(f"ONNX Runtime version: {ort.__version__}")
            print(f"Available providers: {ort.get_available_providers()}")
            print(f"Detected GPUs: {self.num_gpus}")
        else:
            print("ONNX Runtime: NOT INSTALLED")
        
        print(f"Python version: {sys.version}")
        print(f"Model path: {self.model_path}")
        
        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
        
        # Check for GIL status (Python 3.13+)
        if hasattr(sys, '_is_gil_enabled'):
            print(f"GIL enabled: {sys._is_gil_enabled()}")
        
        print("="*70 + "\n")

    def create_session_on_gpu(self, gpu_id: int, session_options: dict = None) -> TimingResult:
        """Create a single ONNX session on specified GPU and time it."""
        thread_id = threading.current_thread().name
        start = time.perf_counter()
        
        try:
            sess_opts = ort.SessionOptions()
            
            # Apply any custom session options
            if session_options:
                if 'intra_op_num_threads' in session_options:
                    sess_opts.intra_op_num_threads = session_options['intra_op_num_threads']
                if 'inter_op_num_threads' in session_options:
                    sess_opts.inter_op_num_threads = session_options['inter_op_num_threads']
                if 'graph_optimization_level' in session_options:
                    sess_opts.graph_optimization_level = session_options['graph_optimization_level']
                if 'enable_profiling' in session_options:
                    sess_opts.enable_profiling = session_options['enable_profiling']
                if 'log_severity_level' in session_options:
                    sess_opts.log_severity_level = session_options['log_severity_level']
            
            providers = [
                ('CUDAExecutionProvider', {'device_id': gpu_id}),
                'CPUExecutionProvider'
            ]
            
            sess = ort.InferenceSession(self.model_path, sess_opts, providers=providers)
            
            end = time.perf_counter()
            
            # Get some info about the session
            extra_info = {
                'inputs': [inp.name for inp in sess.get_inputs()],
                'outputs': [out.name for out in sess.get_outputs()],
                'providers_used': sess.get_providers(),
            }
            
            del sess
            gc.collect()
            
            return TimingResult(
                gpu_id=gpu_id,
                duration_seconds=end - start,
                thread_or_process_id=thread_id,
                start_time=start,
                end_time=end,
                success=True,
                extra_info=extra_info
            )
            
        except Exception as e:
            end = time.perf_counter()
            return TimingResult(
                gpu_id=gpu_id,
                duration_seconds=end - start,
                thread_or_process_id=thread_id,
                start_time=start,
                end_time=end,
                success=False,
                error=str(e)
            )

    def test_sequential_initialization(self) -> List[TimingResult]:
        """Initialize sessions sequentially (baseline)."""
        print("\n[TEST 1] Sequential Session Initialization")
        print("-" * 50)
        
        results = []
        total_start = time.perf_counter()
        
        for gpu_id in range(self.num_gpus):
            print(f"  Initializing session on GPU {gpu_id}...", end=" ", flush=True)
            result = self.create_session_on_gpu(gpu_id)
            results.append(result)
            print(f"Done in {result.duration_seconds:.2f}s")
        
        total_time = time.perf_counter() - total_start
        print(f"\n  Total time (sequential): {total_time:.2f}s")
        print(f"  Average per GPU: {total_time/self.num_gpus:.2f}s")
        
        self.results['sequential'] = {
            'results': results,
            'total_time': total_time
        }
        return results

    def test_threaded_initialization(self) -> List[TimingResult]:
        """Initialize sessions in parallel using threads."""
        print("\n[TEST 2] Threaded Parallel Session Initialization")
        print("-" * 50)
        
        results = []
        total_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [
                executor.submit(self.create_session_on_gpu, gpu_id)
                for gpu_id in range(self.num_gpus)
            ]
            results = [f.result() for f in futures]
        
        total_time = time.perf_counter() - total_start
        
        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"  GPU {r.gpu_id}: {r.duration_seconds:.2f}s ({status})")
        
        print(f"\n  Total wall time (threaded): {total_time:.2f}s")
        print(f"  Sum of individual times: {sum(r.duration_seconds for r in results):.2f}s")
        
        # Calculate overlap
        if results:
            min_start = min(r.start_time for r in results)
            max_end = max(r.end_time for r in results)
            theoretical_parallel = max(r.duration_seconds for r in results)
            actual_parallel = max_end - min_start
            parallelism_ratio = theoretical_parallel / actual_parallel if actual_parallel > 0 else 0
            
            print(f"  Theoretical parallel time: {theoretical_parallel:.2f}s")
            print(f"  Parallelism efficiency: {parallelism_ratio*100:.1f}%")
            
            if parallelism_ratio < 0.5:
                print("\n  ⚠️  LOW PARALLELISM DETECTED - likely GIL or internal locking issue")
        
        self.results['threaded'] = {
            'results': results,
            'total_time': total_time
        }
        return results

    def test_multiprocess_initialization(self) -> List[TimingResult]:
        """Initialize sessions in parallel using processes."""
        print("\n[TEST 3] Multiprocess Parallel Session Initialization")
        print("-" * 50)
        print("  (This bypasses the GIL)")
        
        total_start = time.perf_counter()
        
        # Use spawn to avoid issues with CUDA contexts
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=self.num_gpus, mp_context=ctx) as executor:
            futures = [
                executor.submit(_create_session_in_process, self.model_path, gpu_id)
                for gpu_id in range(self.num_gpus)
            ]
            results = [f.result() for f in futures]
        
        total_time = time.perf_counter() - total_start
        
        for r in results:
            status = "OK" if r['success'] else f"FAILED: {r.get('error', 'Unknown')}"
            print(f"  GPU {r['gpu_id']}: {r['duration_seconds']:.2f}s ({status})")
        
        print(f"\n  Total wall time (multiprocess): {total_time:.2f}s")
        print(f"  Sum of individual times: {sum(r['duration_seconds'] for r in results):.2f}s")
        
        self.results['multiprocess'] = {
            'results': results,
            'total_time': total_time
        }
        return results

    def test_session_options_impact(self):
        """Test impact of various session options on initialization time."""
        print("\n[TEST 4] Session Options Impact on Initialization Time")
        print("-" * 50)
        
        if self.num_gpus == 0:
            print("  Skipping - no GPUs available")
            return
        
        gpu_id = 0  # Test on first GPU only
        
        test_configs = [
            ("Default", {}),
            ("No graph optimization", {
                'graph_optimization_level': ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            }),
            ("Basic optimization only", {
                'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            }),
            ("Single thread", {
                'intra_op_num_threads': 1,
                'inter_op_num_threads': 1
            }),
        ]
        
        results = {}
        for name, options in test_configs:
            print(f"  Testing '{name}'...", end=" ", flush=True)
            
            # Force garbage collection between tests
            gc.collect()
            time.sleep(0.5)
            
            result = self.create_session_on_gpu(gpu_id, options)
            results[name] = result.duration_seconds
            print(f"{result.duration_seconds:.2f}s")
        
        self.results['session_options'] = results
        
        # Find the fastest
        fastest = min(results.items(), key=lambda x: x[1])
        print(f"\n  Fastest option: '{fastest[0]}' at {fastest[1]:.2f}s")

    def test_cuda_provider_options(self):
        """Test CUDA-specific provider options that affect initialization."""
        print("\n[TEST 5] CUDA Provider Options Impact")
        print("-" * 50)
        
        if self.num_gpus == 0 or 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("  Skipping - CUDA not available")
            return
        
        gpu_id = 0
        
        cuda_configs = [
            ("Default CUDA options", {}),
            ("cudnn_conv_algo_search=DEFAULT", {'cudnn_conv_algo_search': 'DEFAULT'}),
            ("cudnn_conv_algo_search=HEURISTIC", {'cudnn_conv_algo_search': 'HEURISTIC'}),
            ("arena_extend_strategy=kSameAsRequested", {'arena_extend_strategy': 'kSameAsRequested'}),
            ("cudnn_conv_use_max_workspace=0", {'cudnn_conv_use_max_workspace': '0'}),
        ]
        
        results = {}
        for name, cuda_opts in cuda_configs:
            print(f"  Testing '{name}'...", end=" ", flush=True)
            
            gc.collect()
            time.sleep(0.5)
            
            try:
                start = time.perf_counter()
                sess_opts = ort.SessionOptions()
                
                providers = [
                    ('CUDAExecutionProvider', {'device_id': gpu_id, **cuda_opts}),
                    'CPUExecutionProvider'
                ]
                
                sess = ort.InferenceSession(self.model_path, sess_opts, providers=providers)
                del sess
                
                duration = time.perf_counter() - start
                results[name] = duration
                print(f"{duration:.2f}s")
                
            except Exception as e:
                results[name] = f"FAILED: {e}"
                print(f"FAILED: {e}")
        
        self.results['cuda_options'] = results

    def profile_session_creation(self):
        """Create a session with profiling enabled to see where time is spent."""
        print("\n[TEST 6] Detailed Profiling of Session Creation")
        print("-" * 50)
        
        if self.num_gpus == 0:
            print("  Skipping - no GPUs available")
            return
        
        gpu_id = 0
        profile_file = f"/tmp/onnx_session_profile_{int(time.time())}"
        
        print(f"  Creating session with profiling enabled...")
        
        sess_opts = ort.SessionOptions()
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = profile_file
        sess_opts.log_severity_level = 0  # Verbose
        
        providers = [
            ('CUDAExecutionProvider', {'device_id': gpu_id}),
            'CPUExecutionProvider'
        ]
        
        start = time.perf_counter()
        sess = ort.InferenceSession(self.model_path, sess_opts, providers=providers)
        init_time = time.perf_counter() - start
        
        # Do a dummy run to capture inference profiling too
        if NP_AVAILABLE:
            try:
                inputs = {}
                for inp in sess.get_inputs():
                    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    if inp.type == 'tensor(float)':
                        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
                    elif inp.type == 'tensor(int64)':
                        inputs[inp.name] = np.random.randint(0, 100, shape).astype(np.int64)
                
                if inputs:
                    _ = sess.run(None, inputs)
            except Exception as e:
                print(f"  (Could not run dummy inference: {e})")
        
        # End profiling
        profile_path = sess.end_profiling()
        print(f"  Session initialization time: {init_time:.2f}s")
        print(f"  Profile saved to: {profile_path}")
        
        # Try to parse and summarize the profile
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                
                print(f"\n  Profile Summary (top 10 by duration):")
                
                # Extract events with duration
                events = []
                for item in profile_data:
                    if isinstance(item, dict) and 'dur' in item:
                        events.append({
                            'name': item.get('name', 'unknown'),
                            'duration_us': item['dur'],
                            'cat': item.get('cat', 'unknown')
                        })
                
                # Sort by duration
                events.sort(key=lambda x: x['duration_us'], reverse=True)
                
                for i, event in enumerate(events[:10]):
                    dur_ms = event['duration_us'] / 1000
                    print(f"    {i+1}. {event['name']}: {dur_ms:.2f}ms ({event['cat']})")
                
                self.results['profile'] = {
                    'profile_path': profile_path,
                    'init_time': init_time,
                    'top_events': events[:10]
                }
                
            except Exception as e:
                print(f"  Could not parse profile: {e}")
        
        del sess
        gc.collect()

    def analyze_and_recommend(self):
        """Analyze results and provide recommendations."""
        print("\n" + "="*70)
        print("ANALYSIS AND RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        # Check for GIL/locking issues
        if 'sequential' in self.results and 'threaded' in self.results:
            seq_time = self.results['sequential']['total_time']
            thread_time = self.results['threaded']['total_time']
            
            # If threaded is not significantly faster than sequential
            if thread_time > seq_time * 0.8:
                recommendations.append(
                    "⚠️  THREADING NOT HELPING: Threaded initialization is not faster than "
                    "sequential. This strongly suggests GIL holding or internal locking.\n"
                    "   → Consider using multiprocessing instead of threading.\n"
                    "   → See test 3 results for multiprocess performance."
                )
        
        # Check if multiprocess helps
        if 'threaded' in self.results and 'multiprocess' in self.results:
            thread_time = self.results['threaded']['total_time']
            mp_time = self.results['multiprocess']['total_time']
            
            if mp_time < thread_time * 0.7:
                recommendations.append(
                    "✓ MULTIPROCESSING HELPS: Using separate processes is significantly faster.\n"
                    "   → The issue is likely Python GIL-related.\n"
                    "   → Implement a process pool for session initialization.\n"
                    "   → Use shared memory (multiprocessing.shared_memory) for data sharing."
                )
        
        # Check session options
        if 'session_options' in self.results:
            opts = self.results['session_options']
            default_time = opts.get('Default', float('inf'))
            
            for name, duration in opts.items():
                if isinstance(duration, float) and duration < default_time * 0.7:
                    recommendations.append(
                        f"✓ SESSION OPTION '{name}' reduces init time significantly.\n"
                        f"   → Default: {default_time:.2f}s vs {name}: {duration:.2f}s\n"
                        f"   → Consider using this option in production."
                    )
        
        # Check CUDA options
        if 'cuda_options' in self.results:
            opts = self.results['cuda_options']
            default_time = opts.get('Default CUDA options', float('inf'))
            
            for name, result in opts.items():
                if isinstance(result, float) and result < default_time * 0.7:
                    recommendations.append(
                        f"✓ CUDA OPTION '{name}' reduces init time.\n"
                        f"   → This is often the biggest factor for slow GPU init.\n"
                        f"   → cudnn_conv_algo_search='EXHAUSTIVE' (default) is very slow.\n"
                        f"   → Use 'DEFAULT' or 'HEURISTIC' instead."
                    )
        
        # General recommendations
        recommendations.append(
            "\n📋 GENERAL RECOMMENDATIONS:\n"
            "   1. For large models, save the optimized model once:\n"
            "      sess_opts.optimized_model_filepath = 'model_optimized.onnx'\n"
            "   2. Consider converting to ORT format for faster loading:\n"
            "      python -m onnxruntime.transformers.convert_to_ort model.onnx\n"
            "   3. Disable cudnn_conv_algo_search=EXHAUSTIVE:\n"
            "      providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'})]\n"
            "   4. For parallel init across GPUs, use multiprocessing with 'spawn' context.\n"
            "   5. Enable verbose logging to see what's happening:\n"
            "      sess_opts.log_severity_level = 0"
        )
        
        for rec in recommendations:
            print(f"\n{rec}")
        
        print("\n" + "="*70)

    def run_all_tests(self):
        """Run all diagnostic tests."""
        self.print_system_info()
        
        if not ORT_AVAILABLE:
            print("ERROR: onnxruntime is not installed. Cannot run tests.")
            return
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found: {self.model_path}")
            return
        
        # Run tests
        self.test_sequential_initialization()
        self.test_threaded_initialization()
        
        try:
            self.test_multiprocess_initialization()
        except Exception as e:
            print(f"  Multiprocess test failed: {e}")
        
        self.test_session_options_impact()
        self.test_cuda_provider_options()
        self.profile_session_creation()
        
        # Analyze and recommend
        self.analyze_and_recommend()
        
        return self.results


def _create_session_in_process(model_path: str, gpu_id: int) -> dict:
    """Function to create session in a separate process."""
    import onnxruntime as ort
    import time
    import os
    
    start = time.perf_counter()
    
    try:
        sess_opts = ort.SessionOptions()
        providers = [
            ('CUDAExecutionProvider', {'device_id': gpu_id}),
            'CPUExecutionProvider'
        ]
        
        sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
        end = time.perf_counter()
        
        result = {
            'gpu_id': gpu_id,
            'duration_seconds': end - start,
            'process_id': os.getpid(),
            'success': True
        }
        
        del sess
        return result
        
    except Exception as e:
        end = time.perf_counter()
        return {
            'gpu_id': gpu_id,
            'duration_seconds': end - start,
            'process_id': os.getpid(),
            'success': False,
            'error': str(e)
        }


def create_dummy_model(output_path: str):
    """Create a simple dummy ONNX model for testing."""
    try:
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model: Y = X * W + B
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 256])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 256])
        
        W = helper.make_tensor('W', TensorProto.FLOAT, [256, 256],
                               np.random.randn(256, 256).astype(np.float32).flatten().tolist())
        B = helper.make_tensor('B', TensorProto.FLOAT, [256],
                               np.random.randn(256).astype(np.float32).flatten().tolist())
        
        matmul = helper.make_node('MatMul', ['X', 'W'], ['matmul_out'])
        add = helper.make_node('Add', ['matmul_out', 'B'], ['Y'])
        
        graph = helper.make_graph([matmul, add], 'test_graph', [X], [Y], [W, B])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
        model.ir_version = 8  # Set compatible IR version

        onnx.save(model, output_path)
        print(f"Created dummy model at: {output_path}")
        return True
        
    except ImportError:
        print("Cannot create dummy model - 'onnx' package not installed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose ONNX Runtime session initialization performance issues'
    )
    parser.add_argument('--model', type=str, 
                        help='Path to ONNX model file')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to test (auto-detect if not specified)')
    parser.add_argument('--create-dummy', action='store_true',
                        help='Create a dummy model for testing')
    
    args = parser.parse_args()
    
    if args.create_dummy:
        dummy_path = '/tmp/dummy_model.onnx'
        if create_dummy_model(dummy_path):
            args.model = dummy_path
    
    if not args.model:
        print("ERROR: Please specify a model path with --model or use --create-dummy")
        print("\nUsage examples:")
        print("  python onnx_session_diagnostics.py --model path/to/model.onnx")
        print("  python onnx_session_diagnostics.py --create-dummy")
        sys.exit(1)
    
    runner = DiagnosticRunner(args.model, args.num_gpus)
    runner.run_all_tests()


if __name__ == '__main__':
    main()
