
__global__  void add(float* x, float* y, float* z, int n) {

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
    
    return 0;
}