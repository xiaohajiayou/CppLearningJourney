
## 知识点覆盖
![alt text](images/image-2.png)
- 算子层：熟悉常用的卷积计算优化方法：gemm、winograd算法
- 编译层：TVM/MLIR/IREE
- 算法层：了解主流AIGC算法模型原理
- 框架层：开源推理框架贡献：DeepSpeed，FasterTransformer、vllm、lmdeploy、ncnn、tensorRT
- 汇编层： arm neon指令、汇编优化、GPU优化

- 熟悉常用图像视觉计算库及深度学习推理计算库：arm compute library
- 系统架构层： 计算图和 OP 的优化，缓存/显存优化，LLM 结构运行时的系统架构。
- 搜推场景推理服务研发：使用ONNX Runtime/TensorRT等框架，深度优化GPU/CPU模型推理效率，打造高吞吐低延迟的模型推理服务。支持大规模稀疏模型的分布式存储、实时更新、低延迟通信，提供行业先进的推理引擎。探索下一代基于大规模稠密参数的推荐服务。
- 大模型推理优化：使用SGLang/vLLM/TensorRT-LLM/Triton-Inference-Server等框架和引擎，部署和优化
- 了解分布式推理、量化、稀疏加速，或对算子、访存、通信优化等有一定经验。

## cuda
### 学习资料
![alt text](images/image.png)

- 书籍
  - 《CUDA并行编程指南》、《cuda c编程权威指南》、《通用图形处理器设计:GPGPU编程模型与架构原理》
  - NVIDA官方的Ducumation（https://docs.nvidia.com/cuda/index.html）
  - 樊哲勇 cuda书籍
  - 知乎bbuf 公众号giantpandacv
  - 知乎onflow
  - b站 nsight compute
- 课程
https://www.bilibili.com/video/BV1kx411m7Fk/?spm_id_from=333.337.search-card.all.click&vd_source=09dab0452e2548023f6f83174148ee0c
### 实战
可以试试用cuda实现reduce/histogram/softmax/gemm/scan/sort之类的算法，尤其是gemm，感觉高性能计算方向大概率会问，知乎上也有很多大佬写的文章可以参考，cuda core和tensor core的实现都有。（如果都实现过了，当我没说...   


## 推理框架
 
- 如果是大模型部署和推理优化的话，感觉还需要了解一点fastertransformer、flashattention和vllm之类的框架
- 推理框架优化（内存优化、分层框架）、算子优化（cvpr最新算子支持，插件实现、异构推理）方向     




### ggml

### llama.cpp
https://www.bilibili.com/video/BV1Ez4y1w7fc/?spm_id_from=333.337.search-card.all.click&vd_source=09dab0452e2548023f6f83174148ee0c
### vllm

https://blog.csdn.net/weixin_42479327/article/details/141496484

![alt text](images/image-1.png)


## 硬件相关
### 计算机体系结构
- 一个是[计算机体系结构](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=%E8%AE%A1%E7%AE%97%E6%9C%BA%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84&zhida_source=entity)。
  这个领域很基础，但对理解AI Infra的性能优化至关重要，尤其是像GPU、[TPU](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=TPU&zhida_source=entity)、[ASIC](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=ASIC&zhida_source=entity)这些硬件的设计思路和架构原理。经典的书有《Computer Architecture: A Quantitative Approach》，然后再结合实际了解NVIDIA的GPU白皮书、CUDA文档，把理论和实践结合起来。
- 第二个是分布式系统和并行计算。
  AI Infra的大模型训练基本都要跑在多机多卡甚至超大规模集群上，分布式计算的核心理念必须掌握，比如数据并行、模型并行、分布式存储、RPC（远程过程调用）等。可以从[《Distributed Systems》](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=%E3%80%8ADistributed+Systems%E3%80%8B&zhida_source=entity)这类经典书入门，再去实践框架，比如[PyTorch](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=PyTorch&zhida_source=entity)的分布式训练（DDP）或者NVIDIA的 NCCL库。你甚至可以深入研究像[Ray](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=Ray&zhida_source=entity)、[Horovod](https://zhida.zhihu.com/search?content_id=700923002&content_type=Answer&match_order=1&q=Horovod&zhida_source=entity)这样的分布式框架。
### gpu架构
- https://zhuanlan.zhihu.com/p/11438556321
- 《通用图形处理器设计:GPGPU编程模型与架构原理》
- CUDA C Programming Guide（https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html）



### 芯片体系结构












## 面经
### cuda
*   cuda graph作用原理，kernel launch流程
*   如何确定blocksize和gridsize
*   什么是default stream，它存在什么问题
*   shared memor的bank conflict及解决方法
*   threadfence的作用
*   如何debug cuda kernel
*   unified memory和zero-copy memory
*   cuda sort如何实现
*   sin函数在哪个硬件计算，这个硬件还能算什么
*   Volat架构特性，ITS
*   3090上单个block能用的shmem最大有多少
*   PTX与SASS的区别
*   GPU性能xx TFLOPS是怎么计算的
*   
### cpp
*   C++虚函数实现机制，单继承、多继承、虚继承的内存布局
*   四种cast
*   三种智能指针
*   函数模板声明与定义能否分离
*   CRTP静态多态
*   vector扩容，resize和reserve
*   单例模式
*   

### 手撕
做推理优化和高性能计算肯定是要懂点cuda，所以大部分的题目都是用cuda实现，一些不太好用cuda实现的如NMS就用c++写了。当然也遇到过一些力扣题目，基本是hot100的范畴，这里不再赘述。

cuda实现：reduction，softmax，matrix transpose，avg pooling，算两堆bbox的iou，大部分情况下都是实现kernel即可，少数情况需要跟cpu对齐。

c++实现：NMS，conv2d，双线性插值，layernorm，单例模式

这里面让我印象比较深刻的是layernorm，用cuda写个layernorm不难，但面试官让我用vadd/vsub/vmul/vdiv等向量指令实现一个layernorm，我人都傻了。一是咱平时写cuda都是SIMT的编程模型，cpu优化是SIMD，这俩写起来有差别；二是没提供sqrt，得自己用牛顿法求，而且还没有比较运算符，浮点数的比较还有一些trick，最后肯定是寄了。

另外就是某大模型公司，要求实现softmax，需要跟cpu版本对齐。

### jd
![alt text](images/image-3.png)
![alt text](images/image-4.png)
![alt text](images/image-5.png)
![alt text](images/image-6.png)


*   熟练掌握GPU CUDA编程
*   追求技术极致，务实，渴望有自己的作品和代表作
*   加分项

*   对芯片体系结构和指令执行pipeline有深刻认知者
*   MLSys方向的知名开源项目贡献者
*   ACM等编程竞赛获奖者