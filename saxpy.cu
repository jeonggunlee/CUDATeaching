#include <stdio.h>
#include <cuda.h>
// CUDA Saxpy 
__global__ void saxpy(int n, float a, float* x, float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    y[idx] = a*x[idx] + y[idx];
}

int main()
{
    int n = 1024*256;
    float a = 0.15;
    float x[n];
    float y[n];
    float h_y[n];
    float* d_x;
    float* d_y;

    // initialize
    for(int i=0; i<n; i++) {
        y[i] = i*0.01 + 0.4;
        h_y[i] = i*0.01 + 0.4;
        x[i] = i*0.02 + 0.2;
    }

    // cudaMalloc ( void** devPtr, size_t size ) 
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));

    // __host__​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    //int blocks = n / 256;
    // int n, float a, float x, float y
    saxpy<<<n / 256, threadPerBlock>>>(n, a, d_x, d_y);

    cudaDeviceSynchronize();
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Correct Function
    for(int i=0; i<n; i++) h_y[i] = a*x[i] + h_y[i];

    // Verification
    float diff;
    float errorSum=0;
    for(int i=0; i<n; i++) {
        diff = y[i] - h_y[i];
        errorSum = errorSum + diff*diff;
    }
    printf("Error Sum = %f\n", errorSum);
    for(int i=0; i<10; i++) {
        printf("Host Results:Device Results %f %f\n", h_y[i], y[i]);
    }

    // __host__ ​ __device__ ​cudaError_t cudaFree ( void* devPtr ) 
    cudaFree(d_x);
    cudaFree(d_y);
}