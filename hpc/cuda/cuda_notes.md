你的观察非常敏锐，`cudaMalloc` 的设计确实与标准 C 语言的 `malloc` 有所不同，这背后涉及 CUDA 编程模型的一些核心特性和设计权衡。让我从几个角度解释这个问题：


### 1. **CUDA 内存分配的复杂性**
标准 C 语言的 `malloc` 只负责在**主机内存**（CPU 内存）中分配空间，并返回指向该内存的指针。而 `cudaMalloc` 要做的事情更复杂：
- **跨设备管理**：`cudaMalloc` 在**GPU 显存**（设备内存）中分配空间，这涉及主机和设备之间的通信和同步。
- **错误处理**：CUDA API 需要返回错误码（如 `cudaError_t`）来指示内存分配是否成功，而函数只能有一个返回值。


### 2. **设计选择：返回值 vs 输出参数**
如果 `cudaMalloc` 像 `malloc` 一样用返回值传递指针，就无法直接返回错误状态。CUDA 团队有两种选择：

#### 方案一：通过返回值传递指针，错误处理靠全局变量
类似标准 C 的 `errno`，CUDA 可以设计为：
```c
void* cudaMalloc(size_t size);  // 返回分配的指针，错误通过 cudaGetLastError() 获取
```
但这种方式有明显缺点：
- 必须手动检查每次调用后的错误状态，容易遗漏。
- 多线程环境下全局错误状态可能被其他线程覆盖。

#### 方案二：通过输出参数传递指针，返回值用于错误码
这正是 CUDA 实际采用的方式：
```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
优点是：
- 强制开发者处理错误（必须检查返回值）。
- 线程安全：每个线程可以独立调用 `cudaMalloc` 而不干扰其他线程。


### 3. **与 CUDA 其他 API 的一致性**
CUDA 中许多涉及资源分配的函数都采用类似模式：
```c
cudaError_t cudaStreamCreate(cudaStream_t* stream);      // 创建流
cudaError_t cudaEventCreate(cudaEvent_t* event);        // 创建事件
cudaError_t cudaMallocHost(void** ptr, size_t size);    // 分配页锁定内存
```
统一的设计让 API 更一致，开发者更容易理解和记忆。


### 4. **如果非要用返回值？**
你当然可以封装一个辅助函数来模拟 `malloc` 的行为：
```c
void* safeCudaMalloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return ptr;
}
```
但这种方式有风险：
- 错误发生时直接退出程序，缺乏灵活性。
- 无法在不终止程序的情况下处理内存分配失败的情况。


### 5. **C++ 中的替代方案**
在 C++ 中，你可以用引用（reference）替代二级指针，让代码更直观：
```cpp
cudaError_t cudaMalloc(void*& devPtr, size_t size);  // 使用引用
```
但 CUDA 的 C API 为了保持与 C 的兼容性，无法使用这种语法。


### 总结
`cudaMalloc` 设计成使用二级指针而非返回值，主要是为了：
1. 同时提供内存指针和错误状态。
2. 强制开发者显式处理错误。
3. 保持与其他 CUDA API 的一致性。
4. 支持多线程环境下的安全操作。

虽然二级指针初看起来有些“拗口”，但这种设计在大型 CUDA 程序中能减少错误，提高代码可靠性。随着 CUDA 生态向 C++ 演进，未来可能会看到更多封装好的 RAII 风格接口（如 `cuda::memory_resource`）来简化内存管理。