# Plan: ONNX Runtime GPU Session Initialization Performance Investigation

## Problem Summary
1. **Multi-threaded GPU session initialization scales linearly** - Threading doesn't help (suspected GIL/locking)
2. **Single session init is very slow** - Large models take ~1 hour (suspected `cudnn_conv_algo_search='EXHAUSTIVE'`)

## Existing Infrastructure
All diagnostic and solution code already exists in `/workspace/tutils/`:
- `check_environment.py` - Environment verification
- `test_gil_detection.py` - Quick GIL confirmation
- `onnx_session_diagnostics.py` - Full diagnostic suite (6 tests)
- `onnx_solutions.py` - Production-ready solutions

---

## Phase 1: Environment Setup

### Step 1.1: Install Dependencies
```bash
pip install onnxruntime-gpu onnx huggingface-hub
```

### Step 1.2: Download Test Model (ResNet50)
ResNet50 is ideal because it has many convolutional layers to trigger `cudnn_conv_algo_search` behavior:
```bash
mkdir -p /workspace/models
huggingface-cli download onnxmodelzoo/resnet50-v2-7 --local-dir /workspace/models/resnet50-v2-7
```
Model will be at: `/workspace/models/resnet50-v2-7/resnet50-v2-7.onnx`

**Alternative** - Export from PyTorch if download fails:
```python
import torch
import torchvision.models as models
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "/workspace/models/resnet50.onnx", opset_version=13)
```

---

## Phase 2: Run Diagnostics

### Step 2.1: Environment Check
```bash
python /workspace/tutils/check_environment.py
```
Verify: CUDA provider available, GPU count detected, onnxruntime-gpu installed

### Step 2.2: Quick GIL Detection Test
```bash
python /workspace/tutils/test_gil_detection.py
```
**Expected:** If threading speedup ~1x but multiprocessing shows ~Nx, GIL is confirmed

### Step 2.3: Full Diagnostic Suite
```bash
python /workspace/tutils/onnx_session_diagnostics.py \
  --model /workspace/models/resnet50-v2-7/resnet50-v2-7.onnx \
  --num-gpus 4
```

**6 Tests Run:**
1. Sequential initialization (baseline)
2. Threaded parallel initialization (tests GIL impact)
3. Multiprocess parallel initialization (bypasses GIL)
4. Session options impact
5. CUDA provider options impact (`cudnn_conv_algo_search`)
6. Detailed profiling

---

## Phase 3: Interpret Results

### Key Metrics to Check

| Comparison | GIL/Locking Issue | Healthy |
|------------|-------------------|---------|
| Threaded vs Sequential | ~Same time | Threaded much faster |
| Multiprocess vs Sequential | Much faster | Much faster |
| Threading efficiency | <30% | >70% |

### CUDA Options Impact

| Setting | Init Speed | Inference Speed |
|---------|------------|-----------------|
| EXHAUSTIVE (default) | Very slow | Optimal |
| HEURISTIC | Fast | Good |
| DEFAULT | Fastest | Acceptable |

**Expected:** HEURISTIC gives 10-100x speedup on init for conv-heavy models

---

## Phase 4: Apply Solutions

### Solution A: Fast CUDA Session (Primary fix for slow init)
```python
from onnx_solutions import create_fast_cuda_session
sess = create_fast_cuda_session("model.onnx", gpu_id=0)
# Key: sets cudnn_conv_algo_search='HEURISTIC'
```

### Solution B: Pre-Optimized Model (One-time optimization)
```python
from onnx_solutions import create_and_save_optimized_model, load_optimized_model
# First time (slow):
create_and_save_optimized_model("model.onnx", "model_optimized.onnx", gpu_id=0)
# Subsequently (fast):
sess = load_optimized_model("model_optimized.onnx", gpu_id=0)
```

### Solution C: Multiprocess Session Manager (Fix for GIL)
```python
from onnx_solutions import MultiprocessSessionManager
manager = MultiprocessSessionManager("model.onnx", gpu_ids=[0,1,2,3])
manager.start()  # Parallel initialization
outputs = manager.run(gpu_id=0, inputs={'input': data})
manager.shutdown()
```

---

## Phase 5: Verification

Re-run diagnostics to confirm improvements:
```bash
python /workspace/tutils/onnx_session_diagnostics.py \
  --model /workspace/models/resnet50-v2-7/resnet50-v2-7.onnx \
  --num-gpus 4
```

Compare before/after timing results.

---

## Critical Files

| File | Purpose |
|------|---------|
| `/workspace/tutils/check_environment.py` | Verify setup |
| `/workspace/tutils/test_gil_detection.py` | Quick GIL test |
| `/workspace/tutils/onnx_session_diagnostics.py` | Full diagnostics |
| `/workspace/tutils/onnx_solutions.py` | Apply solutions |
| `/workspace/issue.md` | Problem documentation |

---

## Expected Outcomes

1. **Confirm GIL holding** during session init (threading vs multiprocessing comparison)
2. **Identify primary bottleneck** (likely `cudnn_conv_algo_search='EXHAUSTIVE'`)
3. **Quantify improvements:**
   - Single session: 10-100x faster with HEURISTIC
   - Parallel init: Nx faster with multiprocessing (N = GPU count)
