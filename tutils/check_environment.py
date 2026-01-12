#!/usr/bin/env python3
"""
Environment Check for ONNX Runtime Diagnostics
===============================================

Run this first to verify your setup is ready for testing.

Usage: python check_environment.py
"""

import sys
import subprocess
import shutil

def check_python():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print("  ⚠️  Python 3.8+ recommended")
        return False
    print("  ✓ Python version OK")
    return True

def check_numpy():
    """Check numpy installation."""
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print("  ✓ NumPy installed")
        return True
    except ImportError:
        print("NumPy: NOT INSTALLED")
        print("  → Install with: pip install numpy")
        return False

def check_onnx():
    """Check onnx installation (optional)."""
    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
        print("  ✓ ONNX installed (can create test models)")
        return True
    except ImportError:
        print("ONNX: NOT INSTALLED (optional)")
        print("  → Install with: pip install onnx")
        print("  → Needed to create dummy test models")
        return False

def check_onnxruntime():
    """Check onnxruntime installation."""
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        has_cuda = 'CUDAExecutionProvider' in providers
        has_tensorrt = 'TensorRTExecutionProvider' in providers
        
        if has_cuda:
            print("  ✓ CUDA provider available (GPU support)")
        else:
            print("  ⚠️  CUDA provider NOT available")
            print("     → For GPU testing, install: pip install onnxruntime-gpu")
        
        if has_tensorrt:
            print("  ✓ TensorRT provider available")
        
        return True, has_cuda
        
    except ImportError:
        print("ONNX Runtime: NOT INSTALLED")
        print("  → Install with: pip install onnxruntime")
        print("  → For GPU support: pip install onnxruntime-gpu")
        return False, False

def check_cuda():
    """Check CUDA installation."""
    print("\n--- CUDA Environment ---")
    
    # Check nvidia-smi
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                print(f"GPUs detected: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")
                
                # Get CUDA version from nvidia-smi
                result2 = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                        '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=10)
                
                return len(gpus)
            else:
                print("nvidia-smi failed")
                return 0
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")
            return 0
    else:
        print("nvidia-smi: NOT FOUND")
        print("  → No NVIDIA GPU detected or drivers not installed")
        return 0

def check_cuda_version():
    """Check CUDA toolkit version."""
    if shutil.which('nvcc'):
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        print(f"CUDA Toolkit: {line.strip()}")
                        return True
        except Exception as e:
            print(f"Error checking nvcc: {e}")
    
    # Try to get from nvidia-smi
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        print(f"CUDA (from driver): {line.strip()}")
                        return True
        except:
            pass
    
    print("CUDA Toolkit: Could not determine version")
    return False

def check_torch_cuda():
    """Check PyTorch CUDA (useful for GPU count detection)."""
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"PyTorch GPU count: {torch.cuda.device_count()}")
        return torch.cuda.is_available()
    except ImportError:
        print("\nPyTorch: NOT INSTALLED (optional, helps with GPU detection)")
        return None

def test_onnx_gpu_init():
    """Quick test of ONNX Runtime GPU initialization."""
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("\n--- GPU Init Test: SKIPPED (no CUDA provider) ---")
            return None
        
        print("\n--- GPU Init Test ---")
        
        # Create a minimal model
        try:
            import onnx
            from onnx import helper, TensorProto
            import numpy as np
            import tempfile
            import time
            
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
            
            # Test GPU init
            print("Testing GPU session creation...")
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            
            providers = [
                ('CUDAExecutionProvider', {'device_id': 0}),
                'CPUExecutionProvider'
            ]
            
            start = time.perf_counter()
            sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
            duration = time.perf_counter() - start
            
            actual_providers = sess.get_providers()
            print(f"  Session created in {duration:.3f}s")
            print(f"  Active providers: {actual_providers}")
            
            if 'CUDAExecutionProvider' in actual_providers:
                print("  ✓ GPU is being used")
            else:
                print("  ⚠️  Fell back to CPU!")
            
            # Clean up
            del sess
            import os
            os.unlink(model_path)
            
            return True
            
        except ImportError:
            print("  Cannot test - 'onnx' package not installed")
            return None
            
    except Exception as e:
        print(f"  GPU init test failed: {e}")
        return False

def main():
    print("="*60)
    print("ONNX RUNTIME DIAGNOSTICS - ENVIRONMENT CHECK")
    print("="*60)
    
    print("\n--- Python Environment ---")
    py_ok = check_python()
    np_ok = check_numpy()
    onnx_ok = check_onnx()
    ort_ok, cuda_provider = check_onnxruntime()
    
    num_gpus = check_cuda()
    check_cuda_version()
    torch_cuda = check_torch_cuda()
    
    gpu_init_ok = test_onnx_gpu_init()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nRequired packages:")
    print(f"  {'✓' if py_ok else '✗'} Python 3.8+")
    print(f"  {'✓' if np_ok else '✗'} NumPy")
    print(f"  {'✓' if ort_ok else '✗'} ONNX Runtime")
    
    print("\nOptional packages:")
    print(f"  {'✓' if onnx_ok else '○'} ONNX (for creating test models)")
    print(f"  {'✓' if torch_cuda is not None else '○'} PyTorch (for GPU detection)")
    
    print("\nGPU Support:")
    print(f"  {'✓' if num_gpus > 0 else '✗'} NVIDIA GPU(s) detected: {num_gpus}")
    print(f"  {'✓' if cuda_provider else '✗'} CUDA Execution Provider")
    print(f"  {'✓' if gpu_init_ok else '✗' if gpu_init_ok is False else '○'} GPU initialization test")
    
    print("\n" + "="*60)
    print("WHAT YOU CAN TEST")
    print("="*60)
    
    if ort_ok and np_ok:
        print("\n✓ You can run: python test_gil_detection.py")
        print("  → Tests GIL behavior with threading vs multiprocessing")
    
    if ort_ok and np_ok and onnx_ok:
        print("\n✓ You can run: python onnx_session_diagnostics.py --create-dummy")
        print("  → Full diagnostics with auto-generated test model")
    
    if ort_ok and np_ok:
        print("\n✓ You can run: python onnx_session_diagnostics.py --model YOUR_MODEL.onnx")
        print("  → Full diagnostics with your actual model")
    
    if num_gpus >= 2 and cuda_provider:
        print("\n✓ You have multiple GPUs - can test parallel GPU initialization")
    elif num_gpus == 1 and cuda_provider:
        print("\n⚠️  Only 1 GPU - parallel GPU init testing limited")
        print("   You can still test session options and CUDA provider options")
    elif num_gpus == 0:
        print("\n⚠️  No GPUs detected - CPU-only testing available")
        print("   For GPU testing, you need NVIDIA GPU(s) with CUDA drivers")
    
    if not ort_ok:
        print("\n❌ Install onnxruntime first:")
        print("   pip install onnxruntime          # CPU only")
        print("   pip install onnxruntime-gpu      # With GPU support")
    
    print("")

if __name__ == '__main__':
    main()
