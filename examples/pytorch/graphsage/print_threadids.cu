#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void print_threadids()
{
    printf("ThreadIdx.x: %d, ThreadIdx.y: %d, ThreadIdx.z: %d, BlockIdx.x: %d, BlockIdx.y: %d, BlockIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);
    print_threadids<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}

