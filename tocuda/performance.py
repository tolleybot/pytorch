import torch
import numpy as np
import timeit
import os

print(torch.__version__)

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


def create_tensors(rows=327680, cols=2000, slice_start=10_000, slice_end=50_000):
    """
    Creates two sliced tensors from NumPy arrays with different memory layouts (Fortran and C-style).

    Args:
        rows (int): Number of rows for the arrays. Default is 327680.
        cols (int): Number of columns for the arrays. Default is 2000.
        slice_start (int): Start index for slicing the tensors. Default is 10,000.
        slice_end (int): End index for slicing the tensors. Default is 50,000.

    Returns:
        tensor1 (torch.Tensor): A contiguous slice from a Fortran-order (column-major) NumPy array.
        tensor2 (torch.Tensor): A contiguous slice from a C-order (row-major) NumPy array.
    """
    # Create NumPy arrays: Fortran-contiguous (column-major) and C-contiguous (row-major)
    np_arr_fortran = np.asfortranarray(np.random.randn(rows, cols).astype(np.float32))
    np_arr_c = np.random.randn(rows, cols).astype(np.float32)

    # Convert NumPy arrays to PyTorch tensors
    tensor1 = torch.from_numpy(np_arr_fortran)[slice_start:slice_end]  # Slice the Fortrans tensor
    tensor2 = torch.from_numpy(np_arr_c)[slice_start:slice_end]        # Slice the C-contiguous tensor

    return tensor1, tensor2


def compare_sizes():

    rows = 327680
    cols = 2000
    slice_start = 10_000
    slice_end = 50_000
    rows_1 = rows + 1
    rows_2 = rows - 1

    tensor1 = np.asfortranarray(np.random.randn(rows_1, cols).astype(np.float32))
    tensor2 = np.asfortranarray(np.random.randn(rows_2, cols).astype(np.float32))
    tensor1 = torch.from_numpy(tensor1)[slice_start:slice_end]
    tensor2 = torch.from_numpy(tensor2)[slice_start:slice_end]

    def move_to_gpu(tensor):
        tensor.cuda()

    number = 1

    move_to_gpu(tensor1)

    move_to_gpu(tensor2)

    execution_time_1 = timeit.timeit(lambda: move_to_gpu(tensor1), number=number)
    average_time_1 = execution_time_1 / number

    execution_time_2 = timeit.timeit(lambda: move_to_gpu(tensor2), number=number)
    average_time_2 = execution_time_2 / number

    # Step 8: Print the results
    print(f"Average time for rows {rows_1}: {average_time_1:.6f} seconds")
    print(f"Average time for rows {rows_2}: {average_time_2:.6f} seconds")


def compare_tensors():
    nc_tensor, cont_tensor = create_tensors(rows=32768)

    print("Is the nc_tensor tensor contiguous after Slice? ", nc_tensor.is_contiguous())
    print("Is the cont_tensor tensor contiguous after Slice? ", cont_tensor.is_contiguous())

    def move_to_gpu(tensor):
        tensor.cuda()

    number = 1

    move_to_gpu(nc_tensor)

    move_to_gpu(cont_tensor)

    # Step 6: Use timeit to time the execution of moving a contiguous tensor to the GPU
    # execution_time = timeit.timeit(lambda: move_to_gpu(tensor_contiguous), number=number)
    # average_time = execution_time / number

    # Step 7: Use timeit to time the execution of moving a non-contiguous tensor to the GPU
    execution_time_nc = timeit.timeit(lambda: move_to_gpu(nc_tensor), number=number)
    average_time_nc = execution_time_nc / number

    execution_time = timeit.timeit(lambda: move_to_gpu(cont_tensor), number=number)
    average_time = execution_time / number

    # Step 8: Print the results
    print(f"Average time taken to move contiguous tensor to GPU: {average_time:.6f} seconds")
    print(f"Average time taken to move non-contiguous tensor to GPU: {average_time_nc:.6f} seconds")


def main():
    compare_sizes()


if __name__ == "__main__":
    main()
