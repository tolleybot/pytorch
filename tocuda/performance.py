import torch
import numpy as np
import timeit
import os
import argparse

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch installation path: {torch.__file__}")

print("Process ID:", os.getpid())


def get_device(verbose=True):
    if torch.cuda.is_available():
        if verbose:
            print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if verbose:
            print("Apple M1/M2 is available. Using MPS.")
        return torch.device("mps")
    else:
        if verbose:
            print("Using CPU.")
        return torch.device("cpu")


def check_page_alignment_issues(tensor):
    page_size = os.sysconf('SC_PAGE_SIZE')
    element_size = tensor.element_size()
    shape = tensor.shape
    strides = tensor.stride()

    total_elements = tensor.numel()
    total_size_bytes = total_elements * element_size

    total_size_page_aligned = (total_size_bytes % page_size == 0)

    if len(shape) >= 2:
        row_stride_elements = strides[0]
        row_stride_bytes = row_stride_elements * element_size
        row_size_bytes = row_stride_bytes  # Size in bytes between the start of row n and row n+1

        row_size_page_aligned = (row_size_bytes % page_size == 0)
    else:
        row_size_bytes = None
        row_size_page_aligned = False

    print(f"Total data size: {total_size_bytes} bytes")
    print(f"Total data size aligns with page size: {total_size_page_aligned}")

    if row_size_bytes is not None:
        print(f"Row size in bytes: {row_size_bytes}")
        print(f"Row size aligns with page size: {row_size_page_aligned}")
    else:
        print("Tensor is not at least 2-dimensional; cannot check row size alignment.")

    # Return a dictionary with the results
    return {
        'total_size_page_aligned': total_size_page_aligned,
        'row_size_page_aligned': row_size_page_aligned
    }


def move_to_gpu(tensor):
    tensor.cuda()


def benchmark_gpu_transfer(tensor, number=10):
    def transfer_and_sync():
        move_to_gpu(tensor)

    if not tensor.is_contiguous():
        print("Python: Tensor is NOT contiguous.")

    execution_time = timeit.timeit(transfer_and_sync, number=number)
    average_time = execution_time / number
    return average_time


def create_tensors(rows, cols):
    # Create Fortran-ordered array (non-contiguous in PyTorch)
    fortran_arr = np.asfortranarray(np.random.randn(rows, cols).astype(np.float32))

    # Create C-ordered array (contiguous in PyTorch)
    regular_arr = np.random.randn(rows, cols).astype(np.float32)

    # Convert both NumPy arrays to PyTorch tensors
    non_contiguous_tensor = torch.from_numpy(fortran_arr)
    contiguous_tensor = torch.from_numpy(regular_arr)

    check_page_alignment_issues(non_contiguous_tensor)

    return non_contiguous_tensor, contiguous_tensor


def compare_sizes(rows, cols=2000):
    assert (
        rows >= 50_000
    ), "Rows must be at least 50,000 to support slicing from 10,000 to 50,000."

    non_contiguous_tensor, contiguous_tensor = create_tensors(rows, cols)
    non_contiguous_slice = non_contiguous_tensor[10_000:50_000]
    contiguous_slice = contiguous_tensor[10_000:50_000]

    iter = 5

   
   # warmup
    _ = benchmark_gpu_transfer(non_contiguous_slice, number=1)
    avg_time_contiguous = benchmark_gpu_transfer(contiguous_slice, number=iter)
    avg_time_non_contiguous = benchmark_gpu_transfer(non_contiguous_slice, number=iter)

    # Step 8: Print the results
   # print(f"Blocking: {non_blocking}")
    print(f"Average time for rows {rows} contiguous: {int(avg_time_contiguous * 1000)} ms")
    print(f"Average time for rows {rows} non-contigous: {int(avg_time_non_contiguous * 1000)} ms")


def compare_sizes2(rows, cols):

    non_contiguous_tensor, contiguous_tensor = create_tensors(rows, cols)

    iter = 1

   # avg_time_contiguous = benchmark_gpu_transfer(contiguous_slice, number=iter)
    avg_time_non_contiguous = benchmark_gpu_transfer(non_contiguous_tensor, number=iter)

    # Step 8: Print the results
   # print(f"Blocking: {non_blocking}")
   # print(f"Average time for rows {rows} contiguous: {int(avg_time_contiguous * 1000)} ms")
    print(f"Average time for rows {rows} non-contigous: {int(avg_time_non_contiguous * 1000)} ms")

# python performance.py --rows 327680 --cols 1000
# Average time for rows 327680 non-contigous: 621 ms
# python performance.py --rows 327681 --cols 1000
# Average time for rows 327681 non-contigous: 477 ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU transfer performance.")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows for the tensor.")
    parser.add_argument("--cols", type=int, required=True, help="Number of columns for the tensor.")
    args = parser.parse_args()

    compare_sizes(args.rows, args.cols)
    compare_sizes(args.rows - 1, args.cols)
    # compare_sizes(327680 - 3)

    # compare_sizes(327680)
    # compare_sizes(327680 + 1)

    # compare_sizes(327680 + 2)
    # compare_sizes(327680 + 3)


if __name__ == "__main__":
    main()
