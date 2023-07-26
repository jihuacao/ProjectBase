#include "test_cuda.hpp"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

int main(int* argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i){
        cudaError = cudaGetDeviceProperties(&deviceProp, i);
    }
    cudaSetDevice(1);
    cudaProfilerStart();
    CTest cTest;
    cTest.Evolution();
    cudaProfilerStop();
    printf("a");
    return 0;
}