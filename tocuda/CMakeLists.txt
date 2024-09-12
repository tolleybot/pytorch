cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(project1)


# Replace the path to the directory where LibTorch was built or downloaded
set(Torch_DIR /workspaces/pytorch/torch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(project1 main.cpp)

#link_directories(/opt/conda/lib)

target_link_libraries(project1
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    CUDA::cublas
    CUDA::cufft
    cusparse cupti)

    # Print the directory for cublas and cufft
    message("CUBLAS directory: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    message("CUFFT directory: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")

if (CUDA_FOUND)
    message("CUDA found")
else()
    message("CUDA not found, doing something alternatively")
endif()

set_property(TARGET project1 PROPERTY CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

# For CUDA support (if you built LibTorch with CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")