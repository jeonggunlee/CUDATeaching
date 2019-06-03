#include <stdio.h>

__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //int i = by * blockDim.y + ty;
    //int j = bx * blockDim.x + tx;

    const int tile_size = 16; // tile size

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
        Bs[ty][tx] = N[b + width * ty + tx];
        __syncthreads();

        //for (int k = 0; k < tile_size; ++k)
        //{
        //    Csub += As[ty][k] *  Bs[k][tx];
        //}
        // Loop Unrolling
        Csub += As[ty][0] * Bs[0][tx];
        Csub += As[ty][1] * Bs[1][tx];
        Csub += As[ty][2] * Bs[2][tx];
        Csub += As[ty][3] * Bs[3][tx];
        Csub += As[ty][4] * Bs[4][tx];
        Csub += As[ty][5] * Bs[5][tx];
        Csub += As[ty][6] * Bs[6][tx];
        Csub += As[ty][7] * Bs[7][tx];
        Csub += As[ty][8] * Bs[8][tx];
        Csub += As[ty][9] * Bs[9][tx];
        Csub += As[ty][10] * Bs[10][tx];
        Csub += As[ty][11] * Bs[11][tx];
        Csub += As[ty][12] * Bs[12][tx];
        Csub += As[ty][13] * Bs[13][tx];
        Csub += As[ty][14] * Bs[14][tx];
        Csub += As[ty][15] * Bs[15][tx];
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
                printf("Error !\n");
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
