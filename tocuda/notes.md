
## Python
In our python example we take two tensors and slice them.  One of the tensors is contiguous, and the other is not

```
np_arr_fortran = np.asfortranarray(np.random.randn(rows, cols).astype(np.float32))
np_arr_c = np.random.randn(rows, cols).astype(np.float32)

# Convert NumPy arrays to PyTorch tensors
tensor1 = torch.from_numpy(np_arr_fortran)[slice_start:slice_end]  
tensor2 = torch.from_numpy(np_arr_c)[slice_start:slice_end]       
```

The result is tensor1 (non-contiguous source) stays non-contiguous
tensor2 is also contiguous


## C++

Matching C++ 

```
    // TensorOptions object with the desired data type
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // Create tensor with fortran memory layout
    torch::Tensor tensor_nc = torch::empty_strided({rows, cols}, {1, rows}, options);
    // In-place operation to fill with random values
    tensor_nc.normal_(); 

 
     auto options2 = torch::TensorOptions().dtype(torch::kFloat32);
    // Create contiguous and non-contiguous tensors
    torch::Tensor tensor_cont = torch::randn({rows, cols}, options2); 

    is_contiguous("tensor_nc", tensor_nc);
    is_contiguous("tensor_cont", tensor_cont);   
   

    // Step 2: Slice both tensors
    torch::Tensor slice_cont = tensor_cont.slice(0, 10000, 50000);  // Contiguous slice
    torch::Tensor slice_nc = tensor_nc.slice(0, 10000, 50000);  // Non-contiguous slice

    is_contiguous("slice_nc", slice_nc);
    is_contiguous("slice_cont", slice_cont);
```
The results are the same as python


# aten/src/ATen/native/TensorShape.cpp, Tensor slice(..)


In the last part of the code.  The same results are observed.
All tensors were Non-quantized, and the non-contiguous tensor return non-contiguous 


```
Tensor result;
if (self.is_quantized()) {
auto quantizer = create_subtensor_quantizer(self, false, start_val, end_val, dim, step);
result = as_strided_qtensorimpl(self, sizes, strides, storage_offset, std::move(quantizer));
std::cout << "Quantized tensor" << std::endl;
} else {
result = self.as_strided(sizes, strides, storage_offset);
std::cout << "Non-quantized tensor" << std::endl;
}
namedinference::propagate_names(result, self);
if (result.is_contiguous()) {
std::cout << "result is contiguous" << std::endl;
} else {
std::cout << "result is not contiguous" << std::endl;
}
  return result;
```

### Performance Profiling Results

#### Tensor Transfer Times:
- **Average time taken to move contiguous tensor to GPU**: 0.0058916 seconds
- **Average time taken to move non-contiguous tensor to GPU**: 0.373816 seconds

#### Profiling Application: `./project1`
```
==4385== Profiling application: ./project1
==4385== Profiling result:
```

| **Type**          | **Time(%)** | **Time**   | **Calls** | **Avg**   | **Min**   | **Max**   | **Name**                                                                                                                                                       |
|-------------------|-------------|------------|-----------|-----------|-----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **GPU activities** | 81.77%      | 175.34 ms  | 3         | 58.445 ms | 58.394 ms | 58.519 ms | [CUDA memcpy HtoD]                                                                                                                                             |
|                   | 18.23%      | 39.101 ms  | 1         | 39.101 ms | 39.101 ms | 39.101 ms | _ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_EEvS4_RKT_EUliE_EEviT1_ |
| **API calls**      | 79.34%      | 3.00605 s  | 1         | 3.00605 s | 3.00605 s | 3.00605 s | `cudaDeviceGetStreamPriorityRange`                                                                                                                             |
|                   | 15.98%      | 605.48 ms  | 1         | 605.48 ms | 605.48 ms | 605.48 ms | `cudaLaunchKernel`                                                                                                                                             |
|                   | 4.61%       | 174.51 ms  | 3         | 58.170 ms | 58.093 ms | 58.208 ms | `cudaMemcpyAsync`                                                                                                                                              |
|                   | 0.04%       | 1.4081 ms  | 3         | 469.38 us | 465.77 us | 471.22 us | `cudaStreamSynchronize`                                                                                                                                        |
|                   | 0.02%       | 925.97 us  | 2         | 462.98 us | 422.54 us | 503.43 us | `cudaMalloc`                                                                                                                                                   |
|                   | 0.00%       | 180.85 us  | 101       | 1.7900 us | 179 ns    | 69.632 us | `cuDeviceGetAttribute`                                                                                                                                         |
|                   | 0.00%       | 60.864 us  | 64        | 951 ns    | 393 ns    | 8.1680 us | `cudaGetDevice`                                                                                                                                                |
|                   | 0.00%       | 15.886 us  | 2         | 7.9430 us | 2.8950 us | 12.991 us | `cudaStreamIsCapturing`                                                                                                                                        |
|                   | 0.00%       | 14.003 us  | 1         | 14.003 us | 14.003 us | 14.003 us | `cuDeviceGetName`                                                                                                                                              |
|                   | 0.00%       | 8.4330 us  | 1         | 8.4330 us | 8.4330 us | 8.4330 us | `cuDeviceGetPCIBusId`                                                                                                                                          |
|                   | 0.00%       | 6.1250 us  | 1         | 6.1250 us | 6.1250 us | 6.1250 us | `cuDeviceTotalMem`                                                                                                                                             |
|                   | 0.00%       | 3.8800 us  | 4         | 970 ns    | 294 ns    | 1.5150 us | `cudaGetLastError`                                                                                                                                             |
|                   | 0.00%       | 1.9190 us  | 3         | 639 ns    | 261 ns    | 1.2350 us | `cuDeviceGetCount`                                                                                                                                             |
|                   | 0.00%       | 1.0630 us  | 3         | 354 ns    | 184 ns    | 521 ns    | `cudaGetDeviceCount`                                                                                                                                           |
|                   | 0.00%       | 1.0260 us  | 2         | 513 ns    | 292 ns    | 734 ns    | `cuDeviceGet`                                                                                                                                                  |
|                   | 0.00%       | 670 ns     | 1         | 670 ns    | 670 ns    | 670 ns    | `cuModuleGetLoadingMode`                                                                                                                                       |
|                   | 0.00%       | 351 ns     | 1         | 351 ns    | 351 ns    | 351 ns    | `cuDeviceGetUuid`                                                                                                                                              |

