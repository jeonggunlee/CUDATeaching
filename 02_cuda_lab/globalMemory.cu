//
// Global Memory Access Time Test
// by Jeong-Gun Lee
//
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void MyKernel(unsigned long long* time, int* globalData)
{
  unsigned int s = 1;
  __shared__ float shared[1024];

  unsigned long long startTime = clock();

  globalData[threadIdx.x*s]++;

  unsigned long long finishTime = clock();

  time[threadIdx.x] = (finishTime - startTime);
}

int main()
{
  unsigned long long time[32];
  unsigned long long* d_time;
  int* globalData;

  cudaMalloc(&d_time, sizeof(unsigned long long)*32);

  for(int i=0; i < 10; i++)
  {
    cudaMalloc(&globalData, sizeof(int)*1024*4);
    MyKernel<<<1, 32>>>(d_time, globalData);
    cudaMemcpy(&time, d_time, 32*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // 14 is the overhead for calling clock
    //cout << "Time: " << (time-14)/32 << endl;
    cout << "Time: " << time[0] << " " << time[1] << " " << time[2] << endl;
    cout << endl;
    cudaFree(d_time);
    cudaDeviceSynchronize();
  }

  cudaDeviceReset();

  return 0;
}
