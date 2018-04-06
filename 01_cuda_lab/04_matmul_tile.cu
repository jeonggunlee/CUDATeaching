# Tiled Version 1

#include <stdio.h>

__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //int i = by * blockDim.y + ty;
    //int j = bx * blockDim.x + tx;

    const int tile_size = 16;

    __shared__ int As[tile_size][tile_size];
    __shared__ int Bs[tile_size][tile_size];

    int aBegin = width * tile_size * by;
    int aEnd   = aBegin + width - 1;
    int aStep  = tile_size;

    int bBegin = tile_size * bx;
    int bStep  = tile_size * width;

    int Csub = 0;
    int a, b;

    for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        As[ty][tx] = M[a + width * ty + tx];
        Bs[tx][ty] = N[b + width * tx + ty];  // Not memory coelescing !!!
        __syncthreads();

        for (int k = 0; k < tile_size; ++k)
        {
            Csub += As[ty][k] *  Bs[k][tx];  // No Bank Conflict
        }
        __syncthreads();
    }

    int c = width * tile_size * by + tile_size * bx;
    P[c + width * ty + tx] = Csub;
}

int main(void)
{
    int i, j, k;
    int size=1024;
    int *h_A, *h_B, *h_C, *h_gC;
    int *d_A, *d_B, *d_C;

    int sizeByte = sizeof(int)*size*size;
    h_A = (int *) malloc(sizeByte);
    h_B = (int *) malloc(sizeByte);
    h_C = (int *) malloc(sizeByte);
    h_gC = (int *) malloc(sizeByte);

    for(i = 0; i < size*size; i++) h_A[i] = 1;
    for(i = 0; i < size*size; i++) h_B[i] = 2;

    printf("Host Computing Statrs !\n");
    for(i = 0; i < size; i++)
        for(j = 0; j < size; j++) {
            h_C[i*size+j] = 0;
            for(k = 0; k < size; k++)
                h_C[i*size+j] += h_A[i*size+k]*h_B[k*size+j];
        }
    printf("Host Computing Finished !\n");
/*
    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++)
          printf("%d ", h_C[i*size+j]);
        printf("\n");
    }
*/
    cudaMalloc(&d_A, sizeByte);
    cudaMalloc(&d_B, sizeByte);
    cudaMalloc(&d_C, sizeByte);

    cudaMemcpy(d_A, h_A, sizeByte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeByte, cudaMemcpyHostToDevice);


    printf("GPU Computing Statrs !\n");
    dim3 blocks(size/16, size/16);
    dim3 threads(16, 16);
    MatrixMul<<<blocks, threads >>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    printf("GPU Computing Finished !\n");
    cudaMemcpy(h_gC, d_C, sizeByte, cudaMemcpyDeviceToHost);
/*
    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++)
          printf("%d ", h_gC[i*size+j]);
        printf("\n");
    }
*/
    for(i = 0; i < size; i++)
        for(j = 0; j < size; j++)
            if( h_C[i*size+j] != h_gC[i*size+j] ) {
     for(i = 0; i < size; i++)
        for(j = 0; j < size; j++)
            if( h_C[i*size+j] != h_gC[i*size+j] ) {
                printf("Error !\n");
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                free(h_A);
                free(h_B);
                free(h_C);
                free(h_gC);
                exit(1);
            }

    printf("Success ! \n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_gC);

    exit(0);

}
