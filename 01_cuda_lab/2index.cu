#include <stdio.h>
__global__ void kernel1( int *a )
{
   int idx = blockIdx.x*blockDim.x + threadIdx.x;
   a[idx] = 7;          // output: 7 7 7 7   7 7 7 7   7 7 7 7   7 7 7 7
}

__global__ void kernel2( int *a )
{
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
   a[idx] = blockIdx.x; // output: 0 0 0 0   1 1 1 1   2 2 2 2   3 3 3 3
}

__global__ void kernel3( int *a )
{
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
   a[idx] = threadIdx.x;        // output: 0 1 2 3   1 2 3 4   0 1 2 3   0 1 2 3
}

int main()
{
  int *host_array;
  int *dev_array;

  host_array = (int *) malloc(sizeof(int)*16);
  cudaMalloc(&dev_array, sizeof(int)*16);
  cudaMemset(dev_array, 0, 16);
  kernel1<<<4, 4>>>(dev_array);
  cudaMemcpy(host_array, dev_array, sizeof(int)*16, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 16; i++) printf(" %d ", host_array[i]);
  printf("\n");

  free(host_array);
  cudaFree(dev_array);
  cudaDeviceReset();
  return 0;
}
