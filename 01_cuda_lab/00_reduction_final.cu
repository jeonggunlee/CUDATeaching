#include <stdio.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
      if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
      if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
      if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
      if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
      if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
      if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, int n) {

      extern __shared__ int sdata[];
      // perform first level of reduction, reading from global memory, writing to shared memory
      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
      unsigned int gridSize = blockSize*2*gridDim.x;
      sdata[tid] = 0;
      while (i < n) {
            sdata[tid] += g_idata[i] + g_idata[i+blockSize];
            i += gridSize;
      }
      __syncthreads();

      // do reduction in shared mem
      if (blockSize >= 512) {
            if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
      }
      if (blockSize >= 256) {
            if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
      }
      if (blockSize >= 128) {
            if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
      }

      if (tid < 32) warpReduce<blockSize>(sdata, tid);

      // write result for this block to global mem
      if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



int main(void)
{
  long int size = 1 << 26;
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

  s = size >>  6;
  int blocks = (s+512-1)/512;
  reduce6<512><<<blocks/2, 512, 512*sizeof(int)>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  printf("The size of array is %ld and it is processed on # of Blocks: %d \n", size, blocks/2);
  s = blocks/2;
  blocks = (s+512-1)/512;
  reduce6<512><<<blocks/2, 512, 512*sizeof(int)>>>(d_odata, d_idata, s);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data, d_idata, sizeof(int), cudaMemcpyDeviceToHost);
  printf("GPU result = %d\n", h_data[0]);

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_data);
}
