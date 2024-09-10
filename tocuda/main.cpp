#include <torch/torch.h>
#include <iostream>
#include <chrono>

// Function to move tensor to GPU and time the operation
double measure_gpu_transfer_time(torch::Tensor tensor, int num_iterations) {
    // Function to move tensor to GPU
    auto move_to_gpu = [](torch::Tensor& tensor) {
        tensor = tensor.to(torch::kCUDA);
        torch::cuda::synchronize();  // Ensure the operation has completed
    };

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        move_to_gpu(tensor);  // Move the tensor to the GPU
    }
    torch::cuda::synchronize();  // Synchronize after the last transfer

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
    int rows = 327680;
    int cols = 2000;
    int num_iterations = 10;

    // Step 1: Create contiguous and non-contiguous tensors
    torch::Tensor tensor_r = torch::randn({rows, cols}, torch::kFloat32);  // Contiguous tensor
    torch::Tensor tensor_f = torch::randn({rows, cols}, torch::kFloat32).transpose(0,1);  // Non-contiguous tensor due to transpose

    is_contiguous("tensor_r", tensor_r);
    is_contiguous("tensor_f", tensor_f);
    

    // Step 2: Slice both tensors
    torch::Tensor slice_r = tensor_r.slice(0, 10000, 50000);  // Contiguous slice
    torch::Tensor slice_f = tensor_f.slice(0, 10000, 50000);  // Non-contiguous slice

    is_contiguous("slice_r", slice_r);
    is_contiguous("slice_f", slice_f);

    // warmup
    measure_gpu_transfer_time(slice_r, 10);

    // Step 4: Measure and print time for moving contiguous tensor to GPU
    double avg_time_contiguous = measure_gpu_transfer_time(slice_r, num_iterations);
    std::cout << "Average time taken to move contiguous tensor to GPU: " << avg_time_contiguous << " seconds" << std::endl;

    // Step 5: Measure and print time for moving non-contiguous tensor to GPU
    double avg_time_non_contiguous = measure_gpu_transfer_time(slice_f, num_iterations);
    std::cout << "Average time taken to move non-contiguous tensor to GPU: " << avg_time_non_contiguous << " seconds" << std::endl;

    return 0;
}