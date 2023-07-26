__global__ void matrixMulGlobalKernel(float* pfMatrixA, float* pfMatrixB, float* pfMatrixC, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    for(int i =0; i < k; i++)
    {
        fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
    }

    pfMatrixC[nRow * n + nCol] = fCVal;
}