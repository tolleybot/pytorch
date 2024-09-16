#include <torch/torch.h>
#include <iostream>
#include <chrono>

// Function to move tensor to GPU and time the operation
double measure_gpu_transfer_time(torch::Tensor tensor, int num_iterations) {
    // Function to move tensor to GPU
    auto move_to_gpu = [](torch::Tensor& tensor) {
        tensor = tensor.to(torch::kCUDA);
     //   torch::cuda::synchronize();  // Ensure the operation has completed
    };

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        move_to_gpu(tensor);  // Move the tensor to the GPU
    }
   // torch::cuda::synchronize();  // Synchronize after the last transfer

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Return average time
    return diff.count() / num_iterations;
}

void is_contiguous(const std::string& name,  torch::Tensor& tensor) {
    if (tensor.is_contiguous()) {
        std::cout << name << " is contiguous" << std::endl;
    } else {
        std::cout << name << " is NOT contiguous" << std::endl;
    }   
}

int main() {
    //int rows = 327680;
    int rows = 32768;
    int cols = 2000;
    int num_iterations = 10;

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
    


    #if 1

    // warmup
    measure_gpu_transfer_time(slice_cont, 10);

    // Step 4: Measure and print time for moving contiguous tensor to GPU
    double avg_time_contiguous = measure_gpu_transfer_time(slice_cont, num_iterations);
    std::cout << "Average time taken to move contiguous tensor to GPU: " << avg_time_contiguous << " seconds" << std::endl;

    // Step 5: Measure and print time for moving non-contiguous tensor to GPU
    double avg_time_non_contiguous = measure_gpu_transfer_time(slice_nc, num_iterations);
    std::cout << "Average time taken to move non-contiguous tensor to GPU: " << avg_time_non_contiguous << " seconds" << std::endl;

    #endif

    return 0;
}