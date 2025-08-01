
## reduce baseline
```c
#define THREAD_PER_BLOCK
__global__ void reduce_v0(float *d_in, float *d_out) {
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t idx = threadIdx.x;
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[idx] = d_in[tid];
    __syncthreads();

    for(int i = 1; i < THREAD_PER_BLOCK; i *=2) {
        if ((idx) % (2 * i) == 0) {
            sdata[i] += sdata[+ i];
        }
        __syncthreads();
    }
    if (idx == 0) d_out[blockIdx.x] = sdata[idx];
}

```