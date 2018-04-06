#include <stdio.h>
__global__ void kernel( int *a )
{
   int ix = blockIdx.x*blockDim.x + threadIdx.x;
   int iy = blockIdx.y*blockDim.y + threadIdx.y;
   int idx = iy * blockDim.x * gridDim.x + ix;

   a[idx] = a[idx] + 1;
}

int main()
{
  int *host_array;
  int *dev_array;

  host_array = (int *) malloc(sizeof(int)*16);
  cudaMalloc(&dev_array, sizeof(int)*16);
  cudaMemset(dev_array, 0, 16);

  dim3 block(2,2);
  dim3 threadPerBlock(2,2);
  kernel<<<block, threadPerBlock>>>(dev_array);
  cudaMemcpy(host_array, dev_array, sizeof(int)*16, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 16; i++) printf(" %d ", host_array[i]);
  printf("\n");

  free(host_array);
  cudaFree(dev_array);
  cudaDeviceReset();
  return 0;
}
