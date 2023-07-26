// maxmul.cu

#include <stdio.h>
#include <cuda.h>

__global__ void vecmul(int *A, int* B, int *C, int size) {
    // Row and Column indexes: 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Are they below the maximum?
    if (col < size && row < size) {
       int result = 0;
       for (int ix = 0; ix < size; ix++) {
          result += A[row * size + ix] * B[ix * size + col];
       }
       C[row * size + col] = result;
    }
}

// maxmul 함수의 구현
extern "C" {
    void maxmul(int *A, int* B, int *C, int size) {

    int total = size * size;
    // Allocate device memory:
    int* gpu_A;
    int* gpu_B;
    int* gpu_C;
    int msize = total * sizeof(int);
    cudaMalloc((void**)&gpu_A, msize);
    cudaMemcpy(gpu_A, A, msize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_B, msize);
    cudaMemcpy(gpu_B, B, msize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_C, msize);

    // Blocks & grids:
    int block_size = 32;
    int grid_size = (size + block_size - 1) / block_size;
    dim3 blocks(block_size, block_size);
    dim3 grid(grid_size, grid_size);

    // Call the kernel:
    vecmul<<<grid, blocks>>>(gpu_A, gpu_B, gpu_C, size);
    cudaDeviceSynchronize();

    // Get the result Matrix:
    cudaMemcpy(C, gpu_C, msize, cudaMemcpyDeviceToHost);

    // Free device matrices
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    }
}

