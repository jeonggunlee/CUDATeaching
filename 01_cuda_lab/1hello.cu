#include <stdio.h>

__global__ void helloCUDA(void)
{
  printf("Hello thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
  helloCUDA<<<3, 4>>>();
  cudaDeviceReset();
  return 0;
}
