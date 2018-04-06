#include <stdio.h>

__global__ void reduce0(int *g_idata, int *g_odata) {

      extern __shared__ int sdata[];
      // each thread loads one element from global to shared mem
      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
      sdata[tid] = g_idata[i];
      __syncthreads();
      // do reduction in shared mem
      for(unsigned int s=1; s < blockDim.x; s *= 2) {
            if (tid % (2*s) == 0) {
                  sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
      }
      // write result for this block to global mem
      if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(void)
{
  long int size = 1 << 20;
  long int s;
  int sizeByte = size*sizeof(int);
  int* h_data = (int*) malloc(sizeByte);

  for(int i = 0; i < size; i++) {
    // h_data[i] = rand() & 0xFF;
    h_data[i] = i % 10;
  }

  long long int sum = 0;
  for(int i = 0; i < size; i++) sum += h_data[i];
  printf("CPU results = %lld \n", sum);

  int* d_idata = NULL;
  int* d_odata = NULL;
  cudaMalloc(&d_idata, sizeByte);
  cudaMalloc(&d_odata, sizeByte);

  cudaMemcpy(d_idata, h_data, sizeByte, cudaMemcpyHostToDevice);

  s = size;
  int blocks = (s+512-1)/512;
  reduce0<<<blocks, 512, 512*sizeof(int)>>>(d_idata, d_odata);
  cudaDeviceSynchronize();
  printf("The size of array is %ld and it is processedon # of Blocks: %d \n", s, blocks);

  s = blocks;
  blocks = (s+512-1)/512;
  reduce0<<<blocks, 512, 512*sizeof(int)>>>(d_odata, d_idata);
  cudaDeviceSynchronize();
  printf("The size of array is %ld and it is processedon # of Blocks: %d \n", s, blocks);

  // Now  # of blocks is 4 --> It is better to do computation on CPU.
  cudaMemcpy(h_data, d_idata, blocks*sizeof(int), cudaMemcpyDeviceToHost);

  sum = 0;
  for(int i = 0; i < blocks; i++) {
    printf("%d ", h_data[i]);
    sum += h_data[i];
  }
  printf("\n");
  printf("GPU results = %lld \n", sum);

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_data);
}
