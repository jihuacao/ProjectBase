#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void AddKernel(int *a, int *b, int *c, int DX);