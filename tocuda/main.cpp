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


/**
 * 
 */
void compare_sizes() {

    int rows_1 = 327680 - 1;
    int rows_2 = 327680 + 1;  
    int cols = 2000;
    int num_iterations = 10;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor1 = torch::empty_strided({rows_1, cols}, {1, rows_1}, options);    
    tensor1.normal_(); 

    torch::Tensor tensor2 = torch::empty_strided({rows_2, cols}, {1, rows_2}, options);    
    tensor2.normal_(); 
    
    torch::Tensor slice1 = tensor1.slice(0, 10000, 50000);  
    torch::Tensor slice2 = tensor2.slice(0, 10000, 50000);  
   
    double avg_time_1 = measure_gpu_transfer_time(slice1, num_iterations);
    std::cout << "Average time taken to for rows: " << rows_1 << " = " << avg_time_1 << " seconds" << std::endl;

    double avg_time_2 = measure_gpu_transfer_time(slice2, num_iterations);
    std::cout << "Average time taken for rows: " << rows_2 << " = " << avg_time_2 << " seconds" << std::endl;

}

void compare_contiguous_vs_non_contiguous() {
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
   
    torch::Tensor slice_cont = tensor_cont.slice(0, 10000, 50000);  // Contiguous slice
    torch::Tensor slice_nc = tensor_nc.slice(0, 10000, 50000);  // Non-contiguous slice

    is_contiguous("slice_nc", slice_nc);
    is_contiguous("slice_cont", slice_cont);

    // warmup
    measure_gpu_transfer_time(slice_cont, 10);

    // Measure and print time for moving contiguous tensor to GPU
    double avg_time_contiguous = measure_gpu_transfer_time(slice_cont, num_iterations);
    std::cout << "Average time taken to move contiguous tensor to GPU: " << avg_time_contiguous << " seconds" << std::endl;

    // Measure and print time for moving non-contiguous tensor to GPU
    double avg_time_non_contiguous = measure_gpu_transfer_time(slice_nc, num_iterations);
    std::cout << "Average time taken to move non-contiguous tensor to GPU: " << avg_time_non_contiguous << " seconds" << std::endl;


}

int main() {
    
    // compare_contiguous_vs_non_contiguous();

    compare_sizes();

    return 0;
}