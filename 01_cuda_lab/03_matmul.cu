// 03 Matrix Mulplication
#include <stdio.h>

__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
    int accu = 0;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    for(int k=0; k<width; k++)
    {
        accu = accu + M[i*width+k]*N[k*width+j];
    }

    P[i*width+j] = accu;
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

    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++)
          printf("%d ", h_C[i*size+j]);
        printf("\n");
    }
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

    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++)
          printf("%d ", h_gC[i*size+j]);
        printf("\n");
    }

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
