
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
## 并行编程
https://www.cnblogs.com/orion-orion/p/15376727.html
### cuda
#### 学习资料
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
#### 实战
可以试试用cuda实现reduce/histogram/softmax/gemm/scan/sort之类的算法，尤其是gemm，感觉高性能计算方向大概率会问，知乎上也有很多大佬写的文章可以参考，cuda core和tensor core的实现都有。（如果都实现过了，当我没说...   


## 推理框架
 
- 如果是大模型部署和推理优化的话，感觉还需要了解一点fastertransformer、flashattention和vllm之类的框架
- 推理框架优化（内存优化、分层框架）、算子优化（cvpr最新算子支持，插件实现、异构推理）方向     

### nndeploy
https://nndeploy-zh.readthedocs.io/zh/latest/discussion/discussion.html


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

### 芯片体系架构
#### cpu架构
- 学习常见的芯片架构，如x86架构、ARM架构等，了解它们的特点、优势及适用场景。
- 学会使用性能分析工具（如Intel VTune、Linux perf等）对指令执行pipeline进行分析，找出瓶颈并进行优化
- CPU基础知识，比如CPU如何读取数据，CPU执行一条计算命令的过程是怎么样的；存储系统的层级结构，比如存储系统的层级结构是怎样的，不同层级的存储有什么差别；IO系统知识，比如了解IO如何读写。
- 除此之外，OS知识和编译器的知识也是必须要掌握的。比如，OS如何进行进程调度，如何进行内存分配，如何访问IO设备，如何提供上层接口（什么是系统调用，系统调用如何被执行）、如何优化执行代码等。
- 熟悉并行编程模型：高性能计算的一个重要特点就是并行计算，因此需要掌握并行编程模型，例如共享内存模型（OpenMP）和分布式内存模型（MPI）等。HIP：可移植异构计算接口，厂商中立的C++编程模型，用于实现高度优化的GPU工作负载、OpenCL。
- 这些模型的学习需要对并发编程、同步、互斥、死锁、调度等概念有深刻的理解。
- 《现代CPU性能分析与优化》(ISBN：9787111719489)
#### gpu架构
- https://zhuanlan.zhihu.com/p/11438556321
- 《通用图形处理器设计:GPGPU编程模型与架构原理》
- CUDA C Programming Guide（https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html）













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
*   





## 待整理


北京大学人工智能专业研一在读，目前在某互联网公司高性能计算岗位实习。相比于大佬们的回答，我觉得我的回答可能更符合当下想通过自学找到高性能计算相关岗位工作的同学们的需求，因此斗胆回答一下。

### 为什么互联网公司需要高性能计算？

其实，高性能计算岗位一直以来都是一个相对小众，甚至可以说是冷门的岗位。在深度学习火爆之前，高性能计算岗位存在的主要存在于硬件厂商，做的是底层适配和计算加速工作，门槛较高且需求较少，所以并不知名。然而，随着深度学习逐渐火爆，如何更好的训练和部署深度学习模型也成为了高性能计算研究的课题，而对这一课题的深入探索恰恰是在这一个模型越来越大，对算力需求越来越高的时代中所需要的。尤其是互联网公司更需要这一类人才将算法工程师产出的算法落到实处。因此，近些年来互联网公司对高性能计算的需求越来越高。

### 互联网公司中的高性能计算岗位都负责什么任务？

主要分为三类

1.  硬件相关

包括芯片软件栈的开发，深度学习图编译器的开发，各种[CPU](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=CPU&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiJDUFUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo1NzExNTA5MTUsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.PK3djZmnc1SX_coVrXQl13G0IolJHKy5OD7otW1EIA0&zhida_source=entity)、[GPU](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=GPU&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiJHUFUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo1NzExNTA5MTUsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.QLC47AiPlcca2Ay0tZ81zkBtjKzjTyENj11siYT4OP4&zhida_source=entity)、[NPU](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=NPU&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiJOUFUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo1NzExNTA5MTUsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.RI59yFio77ew1wKg0nrA6THTL8Yv0fxoh5hutteRwoY&zhida_source=entity)适配到特定算法和特定领域的加速工作等等。

2\. 框架相关

包括一些算法中台框架的开发和维护，算法的工程化等等。

3\. 深度学习相关

包括模型在特定平台上的部署优化，尤其是最近大模型火爆后，对大模型的如何训练和部署也纳入了高性能计算岗位负责的范围。

### 高性能计算学习路线？

在上文作者概述了互联网公司高性能计算岗位的相关信息，下面来详细回答一下问题，即高性能计算的学习路线。作者本人是完全自学的高性能计算相关知识，简单的对学习过的知识做了一下总结，具体包括以下几类：

1.  编程语言

高性能计算主要涉及的编程语言为c++和python，其中c++因为其无可比拟的速度优势是高性能计算工程师所必须熟练掌握的编程语言。对应的，高性能计算工程师需要掌握计算机结构与原理、操作系统、编译原理等等计算机相关基础知识。

2\. [并行编程语言](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=%E5%B9%B6%E8%A1%8C%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiLlubbooYznvJbnqIvor63oqIAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo1NzExNTA5MTUsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.NnwhEpIkkDkaMXZKySzOGjr9epPN6qLlH0MGZ2NOszc&zhida_source=entity)

因为各种并行是对程序加速的必要手段之一，所以高性能计算工程师需要掌握一些并行编程手段，例如cuda以及[openMP](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=openMP&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiJvcGVuTVAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo1NzExNTA5MTUsImNvbnRlbnRfdHlwZSI6IkFuc3dlciIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.omfxULnNg6AE4fo-NWZIuA1aBeegy71FFFH2bql6G4M&zhida_source=entity)等。对应的，高性能计算工程师需要掌握CPU/GPU/NPU等芯片的架构以及相关知识，了解如何针对不同硬件进行加速。

3\. 模型训练部署优化

针对目前深度学习模型对算力要求越来越高的趋势，高性能计算工程师需要掌握深度学习相关知识，同时熟练使用模型部署工具（例如[TensorRT](https://zhida.zhihu.com/search?content_id=571150915&content_type=Answer&match_order=1&q=TensorRT&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDUzODkyNDUsInEiOiJUZW5zb3JSVCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjU3MTE1MDkxNSwiY29udGVudF90eXBlIjoiQW5zd2VyIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.eemuDP74lOs9lfY50sELqRbIM_OzmfIobDGjFXaw8KU&zhida_source=entity)）以及了解分布式训练的方法，便于训练无法使用单机单卡训练的大模型。

针对上述知识，作者吐血整理了所有高性能计算相关的知识，上传到了作者的github中，具体如下图所示：

![](https://pic1.zhimg.com/v2-fab35e7a2e92dd77e84ad51a15f6a414_r.jpg?source=1def8aca)

具体链接为：[高性能计算相关知识](https://link.zhihu.com/?target=https%3A//github.com/Eddie-Wang1120/HPC-Learning-Notes)

同时，作者还将高性能计算必备知识之一，cuda编程相关知识的书籍和书籍笔记也上传到了github上，具体如下图所示：

![](https://picx.zhimg.com/v2-644ab598ebca202ce87866a8b045e4bf_r.jpg?source=1def8aca)

具体链接为：[cuda编程相关知识](https://link.zhihu.com/?target=https%3A//github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)

还有，相比于深度学习算法的数学原理与统计学实现，高性能计算岗位更需要了解的是深度学习框架的内部构造和代码实现，便于针对不同场景灵活的使用和开发深度学习框架。作者将自己学习流行的pytorch框架源码的心路历程总结并上传到了知乎上，以供大家共同学习。

具体链接为：

[](https://zhuanlan.zhihu.com/p/537969293)

[](https://zhuanlan.zhihu.com/p/538818127)

[](https://zhuanlan.zhihu.com/p/575297251)

以上内容都在持续更新中，欢迎大家浏览，如果有帮助的话请收藏点赞star！谢谢大家！

![](https://picx.zhimg.com/v2-7b6732d4e4100a54e10fc46f887b8bfc_l.jpg?source=f2fdee93)

