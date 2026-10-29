
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloCUDA()
{
    printf("Hello from GPU! Thread = %d\n", threadIdx.x);
}

int main()
{
    helloCUDA<<<1, 8>>>();

    cudaDeviceSynchronize();

    return 0;
}
