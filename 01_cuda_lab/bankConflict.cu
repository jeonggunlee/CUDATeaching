#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void MyKernel(unsigned long long* time)
{
  unsigned int s = 1;
  __shared__ float shared[1024];

  unsigned long long startTime = clock();   // Starting point of measure
               // For the use of clock() in kernel function,
               // please check https://github.com/zchee/cuda-sample/blob/master/0_Simple/clock/clock.cu

  shared[threadIdx.x*s]++;
      // when s = 1 --> no bank conflict
      //      s = 2 --> two way bank conflict
      //      s = 4 --> four way bank conflict
      // To see a broadcasting case, use "shared[0]++"

  unsigned long long finishTime = clock();  // Ending point of measure
  time[threadIdx.x] = (finishTime - startTime);
}

int main()
{
  unsigned long long time[32];
  unsigned long long* d_time;

  cudaMalloc(&d_time, sizeof(unsigned long long)*32);

  for(int i=0; i < 10; i++)   // Ten time Sampling
  {
    MyKernel<<<1, 32>>>(d_time);
    cudaMemcpy(&time, d_time, 32*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cout << "Time: " << time[0] << " " << time[1] << " " << time[2] << endl;
  }

  cudaFree(d_time);
  cudaDeviceReset();

  return 0;
}
