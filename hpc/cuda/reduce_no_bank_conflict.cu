
__global__ void reduce(float* d_in, float* d_out) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float sdata[blockDim.x];
    sdata[tid] = d_in[index];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }

}

int main() {

    const int N = 1024 * 1024 * 1024;
    const int size = N * sizeof(float);
    dim3 blockSize.x = 256;
    dim3 gridSize.x = N / blockSize.x;
    const int out_size = gridSize.x * sizeof(float);
    float *in, *out;
    in = (float*)malloc(size);
    out = (float*)malloc(out_size);
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, out_size);
    for (int i = 0; i < N; i++) {
        in[i] = 1.0f;
    }
    cudaMemcpy((void*)d_in, (void*)in, cudaMemcpyHostToDevice);

    reduce<<<gridSize, blockSize>>>(d_in, d_out);
    cudaMemcpy((void*)out, (void*)d_out, cudaMemcpyDeviceToHost);

    float* s = (float*)malloc(out_size);
    for (int i = 0; i < gridSize.x; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < blockSize.x; j ++) {
            tmp += in[i * blockSize.x + j];
        }
        s[i] = tmp;
    }

    for(int i = 0; i < gridSize.x; i++) {
        if (out[i] - s[i] > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, s[i], out[i]);
            break;
        } else {
            printf("Success at index %d: expected %f, got %f\n", i, s[i], out[i]);
        }
    }

    

    returen 0;
}