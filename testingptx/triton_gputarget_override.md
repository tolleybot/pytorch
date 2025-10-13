# Using Triton's GPUTarget to Override Compilation Architecture

## The Discovery

Triton supports explicitly specifying the target GPU architecture via `GPUTarget`:

```python
from triton.compiler import GPUTarget

# Instead of auto-detecting from current GPU
target = GPUTarget("cuda", 70, 32)  # sm_70, 32 warps
```

This could potentially allow compiling for sm_70 (V100) even when running on A100 (sm_80).

## The Potential Solution

If we can force Triton to generate sm_70 PTX regardless of the current GPU, then:

1. **Compile on A100:**
   - Override Triton target: `GPUTarget("cuda", 70, 32)`
   - Triton generates sm_70 PTX (not sm_80)
   - PTX contains only sm_70 instructions
   - Can compile SASS for sm_70, 75, 80, 86, 89, 90

2. **Run on V100:**
   - Fatbin contains sm_70 SASS
   - Works! ✅

3. **Run on H100:**
   - Fatbin contains sm_90 SASS (or uses sm_70 PTX)
   - Works! ✅

## How to Implement This

### Challenge: Where is Triton Invoked?

PyTorch Inductor calls Triton internally. We need to find where to inject the target override.

### Possible Approaches

#### Approach 1: Environment Variable

Check if Triton respects an environment variable:

```python
import os
os.environ['TRITON_TARGET_SM'] = '70'  # Force sm_70 target
```

This is the easiest but may not exist.

#### Approach 2: Monkey-patch torch.cuda.get_device_capability

Force PyTorch to report the wrong GPU to Triton:

```python
import torch

# Save original
_original_get_device_capability = torch.cuda.get_device_capability

def _override_get_device_capability(device=None):
    """Force Triton to think we're on V100 (sm_70)"""
    return (7, 0)  # sm_70

# Apply patch
torch.cuda.get_device_capability = _override_get_device_capability

# Now compile model
# Triton will think it's on sm_70
torch._inductor.aoti_compile_and_package(...)

# Restore
torch.cuda.get_device_capability = _original_get_device_capability
```

**Pros:**
- Simple, no Inductor changes needed
- Works from user code

**Cons:**
- Hacky
- May affect other code that checks GPU capability
- Could break assumptions elsewhere

#### Approach 3: Add Inductor Config Option

Add a new config option to Inductor:

```python
# In torch/_inductor/config.py
aot_inductor.target_compute_capability = None  # Default: auto-detect

# Usage:
torch._inductor.aoti_compile_and_package(
    exported_program,
    inductor_configs={
        "aot_inductor.target_compute_capability": "70",  # Force sm_70
        "aot_inductor.emit_multi_arch_kernel": True,
    },
)
```

Then modify Inductor's Triton codegen to use this config.

**Pros:**
- Clean, official API
- Self-documenting

**Cons:**
- Requires modifying PyTorch Inductor
- More complex to implement

#### Approach 4: Hook into Triton Compilation

Find where Inductor calls Triton compiler and inject GPUTarget:

```python
# Somewhere in torch/_inductor/codegen/triton.py
from triton.compiler import GPUTarget

def compile_triton_kernel(...):
    target_sm = config.aot_inductor.get("target_compute_capability")
    if target_sm:
        major = int(target_sm) // 10
        minor = int(target_sm) % 10
        target = GPUTarget("cuda", major * 10 + minor, 32)
        # Pass target to Triton compilation
    else:
        # Use auto-detected
        target = None
```

**Pros:**
- Proper solution
- Uses Triton's official API

**Cons:**
- Requires understanding Inductor's Triton integration
- More invasive changes

## Testing the Monkey-Patch Approach

Let's try the simplest approach first:

```python
import torch
import os

# Method 1: Try environment variable
os.environ['TRITON_TARGET_SM'] = '70'

# Method 2: Monkey-patch
_original = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda device=None: (7, 0)

try:
    # Compile model
    torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path="model.pt2",
        inductor_configs={
            "aot_inductor.emit_multi_arch_kernel": True,
        },
    )
finally:
    # Restore
    torch.cuda.get_device_capability = _original
```

## Potential Issues

### Issue 1: Kernel Performance

If you force sm_70 target on A100:
- Triton generates sm_70 code
- Misses sm_80-specific optimizations
- Performance may be suboptimal on A100

But it would still work correctly, just not optimally.

### Issue 2: Feature Availability

Some Triton kernels may use features only available on newer architectures:
- Tensor cores (sm_70+)
- TMA (sm_90+)
- etc.

If you target sm_70, these features might be unavailable or emulated.

### Issue 3: Triton May Still Auto-Detect

Even if you patch `get_device_capability`, Triton might directly query the GPU via CUDA driver.

## Recommended Implementation Plan

### Phase 1: Experiment (Quick Test)

```python
# In your compile script
import torch

# Monkey-patch approach
original_cap = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda device=None: (7, 0)

try:
    # Compile on A100, but force sm_70 target
    torch._inductor.aoti_compile_and_package(...)

    # Check if it actually generated sm_70 PTX
    # by examining the .ptx files or fatbin
finally:
    torch.cuda.get_device_capability = original_cap
```

Test:
1. Compile on A100
2. Run on V100
3. See if it works!

### Phase 2: Proper Solution (If Experiment Works)

1. Add config option to Inductor
2. Modify Triton codegen to respect config
3. Submit PR to PyTorch

### Phase 3: Update Architecture Filtering

If this works, update the filtering logic:

```python
# OLD (current):
compatible_archs = [
    arch for arch in target_archs
    if int(arch) >= current_arch_int  # Can't include older archs
]

# NEW (with target override):
target_arch_int = config.get("target_compute_capability", current_arch_int)
compatible_archs = [
    arch for arch in target_archs
    if int(arch) >= target_arch_int  # Use explicit target, not current GPU
]
```

This would allow:
- Compile on A100 with target=70
- Generate for sm_70, 75, 80, 86, 89, 90
- Work on all GPUs!

## The Big Question

**Does this actually work?**

You need to test whether:
1. Triton respects the overridden `get_device_capability()`
2. The generated PTX actually targets sm_70
3. The sm_70 PTX compiles successfully
4. The resulting model works on V100

## Quick Test Script

```python
#!/usr/bin/env python3
import torch
import os

# Check current GPU
print(f"Actual GPU: {torch.cuda.get_device_name(0)}")
major, minor = torch.cuda.get_device_capability()
print(f"Actual capability: sm_{major}{minor}")

# Monkey-patch to force sm_70
print("\nForcing sm_70 target...")
torch.cuda.get_device_capability = lambda device=None: (7, 0)

# Verify patch worked
major, minor = torch.cuda.get_device_capability()
print(f"Patched capability: sm_{major}{minor}")

# TODO: Now compile model and check generated PTX
```

## Conclusion

**This is a promising approach** that could potentially solve the backward-compatibility limitation!

**Next steps:**
1. Test the monkey-patch approach
2. Examine generated PTX to verify it targets sm_70
3. Test cross-GPU compatibility
4. If successful, implement proper Inductor config option

This could be the key to true multi-arch support without the "compile on oldest GPU" restriction!
