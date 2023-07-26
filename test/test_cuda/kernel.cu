#include "kernel.cuh"
#include <cuda_tool.h>


__global__ void Add(int *a, int *b, int *c, int DX)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= DX) return;

    c[f] = a[f] + b[f];

}

void AddKernel(int *a, int *b, int *c, int DX)
{
    dim3 dimBlock = (32);
    dim3 dimGrid = ((DX + 32 - 1) / 128);
    Add << <dimGrid, dimBlock >> > (a, b, c, DX);
    cudaDeviceSynchronize();
}