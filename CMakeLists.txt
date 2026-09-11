cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(blur blur.cu)

target_compile_options(blur PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "/Zc:preprocessor" >)