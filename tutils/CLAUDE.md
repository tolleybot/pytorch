# CLAUDE.md - ONNX Runtime Session Initialization Performance Investigation

## Problem Summary

**Reported by:** Casey

**Issue:** When running PyTorch inference jobs and loading ONNX models onto different GPUs on the same node (one model/session per GPU), the time to initialize all ONNX sessions scales linearly with the number of GPUs, even when initializing in separate threads.

**Two related problems:**
1. **Parallel initialization doesn't parallelize** - Threading doesn't help; total time = sum of individual times
2. **Single session init is very slow** - Large models can take ~1 hour to initialize a single session

**Suspected causes:**
- Python GIL being held during session creation
- Internal locking within ONNX Runtime
- CUDA/cuDNN initialization overhead (especially `cudnn_conv_algo_search`)
- Graph optimization overhead

---

## What We Know From Research

### GIL/Threading Issue
- ONNX Runtime's pybind11 bindings may hold the GIL during session initialization
- GitHub issue #26780 confirms the module doesn't declare itself as GIL-safe
- Parallel execution mode doesn't support CUDA Execution Provider
- **Solution:** Use multiprocessing with `spawn` context instead of threading

### Slow Single-Session Initialization
- Primary culprit: `cudnn_conv_algo_search='EXHAUSTIVE'` (default) tests every cuDNN algorithm
- For models with convolutions, this can take enormous time
- First session creation includes CUDA context initialization overhead
- Graph optimization runs during session creation

### Key ONNX Runtime Settings
```python
# Fast CUDA options
providers = [
    ('CUDAExecutionProvider', {
        'device_id': gpu_id,
        'cudnn_conv_algo_search': 'HEURISTIC',  # or 'DEFAULT' - NOT 'EXHAUSTIVE'
        'cudnn_conv_use_max_workspace': '0',
    }),
    'CPUExecutionProvider'
]

# Session options for faster init
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # if pre-optimized
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
sess_opts.log_severity_level = 0  # Verbose for debugging

# Save optimized model (do once)
sess_opts.optimized_model_filepath = 'model_optimized.onnx'
```

---

## Project Files

```
.
├── CLAUDE.md                      # This file
├── check_environment.py           # Verify system setup before testing
├── test_gil_detection.py          # Quick test: is GIL the problem?
├── onnx_session_diagnostics.py    # Full diagnostic suite
├── onnx_solutions.py              # Implemented solutions and workarounds
```

### File Descriptions

**check_environment.py**
- Run first to verify ONNX Runtime, CUDA, GPU availability
- No model file required
- Usage: `python check_environment.py`

**test_gil_detection.py**
- Quick test to determine if GIL holding causes linear scaling
- Compares threading vs multiprocessing performance
- No model file required (uses synthetic workloads)
- Usage: `python test_gil_detection.py`

**onnx_session_diagnostics.py**
- Comprehensive diagnostic suite
- Tests: sequential init, threaded init, multiprocess init, session options, CUDA options, profiling
- Provides analysis and recommendations
- Usage: 
  - `python onnx_session_diagnostics.py --model path/to/model.onnx`
  - `python onnx_session_diagnostics.py --create-dummy` (creates test model)
  - `python onnx_session_diagnostics.py --model path/to/model.onnx --num-gpus 4`

**onnx_solutions.py**
- Implemented solutions ready to use:
  - `create_fast_cuda_session()` - Optimized CUDA provider options
  - `create_and_save_optimized_model()` - Pre-optimize and save
  - `load_optimized_model()` - Load pre-optimized model fast
  - `MultiprocessSessionManager` - Manage sessions across GPUs with separate processes
  - `SharedMemoryBatchManager` - Efficient cross-process data sharing
  - `run_with_io_binding()` - Fast GPU inference with IO binding
  - `create_fast_init_session()` - Minimal init time (trades inference speed)

---

## Diagnostic Workflow

### Step 1: Environment Check
```bash
python check_environment.py
```
Verify: Python 3.8+, numpy, onnxruntime-gpu, CUDA provider available, GPU count

### Step 2: Quick GIL Test
```bash
python test_gil_detection.py
```
Expected result: If ONNX Runtime speedup with threads is ~1x but multiprocessing shows ~Nx speedup (where N = number of tasks), then GIL/locking is confirmed.

### Step 3: Full Diagnostics
```bash
python onnx_session_diagnostics.py --model /path/to/actual/model.onnx
```
This runs 6 tests:
1. Sequential initialization (baseline)
2. Threaded parallel initialization
3. Multiprocess parallel initialization
4. Session options impact
5. CUDA provider options impact
6. Detailed profiling

### Step 4: Apply Solutions
Based on diagnostic results, implement fixes from `onnx_solutions.py`

---

## Most Likely Solutions (In Order of Impact)

### 1. Change cudnn_conv_algo_search (Biggest impact for slow init)
```python
providers = [
    ('CUDAExecutionProvider', {
        'device_id': gpu_id,
        'cudnn_conv_algo_search': 'HEURISTIC',  # Instead of 'EXHAUSTIVE'
    }),
    'CPUExecutionProvider'
]
```

### 2. Use Multiprocessing Instead of Threading
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ctx = multiprocessing.get_context('spawn')  # Important: use spawn for CUDA
with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
    futures = [executor.submit(init_session_on_gpu, gpu_id) for gpu_id in range(num_gpus)]
    results = [f.result() for f in futures]
```

### 3. Pre-optimize Model (Do Once, Load Fast)
```python
# First time: create and save optimized model
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.optimized_model_filepath = 'model_optimized.onnx'
sess = ort.InferenceSession('model.onnx', sess_opts, providers=providers)

# Subsequently: load with optimization disabled
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession('model_optimized.onnx', sess_opts, providers=providers)
```

### 4. Use Shared Memory for Data Sharing Across Processes
```python
from multiprocessing import shared_memory
import numpy as np

# Create shared memory for batch data
shm = shared_memory.SharedMemory(create=True, size=batch_array.nbytes)
shared_array = np.ndarray(batch_shape, dtype=np.float32, buffer=shm.buf)
shared_array[:] = batch_data[:]

# In worker process: attach to shared memory by name
shm = shared_memory.SharedMemory(name=shm_name)
shared_array = np.ndarray(batch_shape, dtype=np.float32, buffer=shm.buf)
```

---

## Hardware Requirements

| Test | Minimum Hardware | Ideal Hardware |
|------|------------------|----------------|
| GIL detection | Any CPU | Any CPU |
| Single GPU options | 1 NVIDIA GPU | 1 NVIDIA GPU |
| Parallel init testing | 2+ NVIDIA GPUs | Same as production |
| Full reproduction | Match production | Match production |

**Software (Native):**
```bash
pip install onnxruntime-gpu numpy onnx
# Ensure CUDA version matches onnxruntime-gpu requirements
nvidia-smi  # Check CUDA version
```

---


## Key References

- ONNX Runtime GitHub Issues:
  - #26780 - GIL warning on Python 3.13t
  - #19022 - Session creation takes too long
  - #9990 - Long init on RTX GPUs
  - #5957 - Model loading slow with onnxruntime-gpu
- ONNX Runtime Docs:
  - https://onnxruntime.ai/docs/performance/tune-performance/threading.html
  - https://onnxruntime.ai/docs/api/python/api_summary.html
  - https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

---

## Notes for Development

- Always use `multiprocessing.get_context('spawn')` with CUDA to avoid context issues
- Profile with `sess_opts.enable_profiling = True` to see where time is spent
- Verbose logging: `sess_opts.log_severity_level = 0`
- The first CUDA session creation is always slower (CUDA context init)
- Test with actual production model for accurate timings
- `cudnn_conv_algo_search='EXHAUSTIVE'` can take hours for large models with many conv layers

---

## TODO / Investigation Items

- [ ] Run environment check on target hardware
- [ ] Run GIL detection test
- [ ] Run full diagnostics with actual model
- [ ] Test `cudnn_conv_algo_search` options impact
- [ ] Test pre-optimized model loading
- [ ] Implement multiprocess session manager if needed
- [ ] Benchmark multiprocess with shared memory for batch data
- [ ] Consider ORT format conversion for faster loading
