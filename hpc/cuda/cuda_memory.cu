
__global__  void add(float* x, float* y, float* z, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        z[tid] = x[tid] + y[tid];
    }
} 


int main() {
    const int n = 1e9;
    const size = n * sizeof(float) * n;
    float *x, *y, *z;
    x = (float*)malloc(sizeof(float) * n);
    y = (float*)malloc(sizeof(float) * n);
    z = (float*)malloc(sizeof(float) * n);

    for(int i = 0; i < n; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMalloc((void**)&d_z, size);

    cudaMemcpy((void*)d_x, (void*)x, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, size, cudaMemcpyHostToDevice);

    dim3 blocksize(256);
    dim3 gridsize((n + blocksize.x - 1) / blocksize.x);
    add<<<gridsize, blocksize>>>(d_x, d_y, d_z, n);

    cudaMemcpy((void*)z, (void*)d_z, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) {
        if (z[i] - 30 > 1e-5) {
            printf("Error at index %d: expected 30, got %f\n", i, z[i]);
            break;
        }
    }

    return 0;
}