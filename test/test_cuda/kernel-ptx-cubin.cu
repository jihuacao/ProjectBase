__global__ void Add(int *a, int *b, int *c, int DX)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= DX) return;

    c[f] = a[f] + b[f];

}