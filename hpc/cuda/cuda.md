## 学习路线
### cuda环境配置
### cuda语法了解
### reduce优化
### 硬件补充
英伟达讲座
cs336
### 调优手段补充

### leetcuda

## 资料
最近工作主要集中在[目标检测算法](https://so.csdn.net/so/search?q=%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95&spm=1001.2101.3001.7020)部署方面，在树莓派4B和NVIDIA GPU平台上做了一些内容，个人觉得GPU多核计算对于深度学习的加持作用意义重大，而NVIDIA出品的软硬件是GPU多核计算的标杆，那就学吧！

（一）CUDA技术路线图

下图是本人梳理的CUDA技术路线图，如果刚入门的话可以按照这个顺序取学习，有了整体的了解之后再逐步深入，每一块展开内容都特别多，有针对性的学习更有效率，下面针对脑图对每一块内容做简要的介绍（复杂介绍的不会）。

![](https://i-blog.csdnimg.cn/blog_migrate/3f49bb835f9d2c115db8618e8adae8d1.jpeg)

（二）CUDA基础知识介绍

1.硬件基础知识：

这一块主要是涉及的知识是CPU与GPU的硬件构成区别，CPU具有强大的控制单元和少量的运算单元，适合逻辑复杂的程序运行，而GPU具有大量的运算核心和核心控制单元，适合大规模的简单逻辑运算。

CUDA优化的本质是对计算资源的分配与调用，需要了解自己常用的硬件平台的GPU的算力及硬件构成，比如多核处理（SM）器个数，流处理器（SP）个数，再进一步线程可调用寄存器个数等等；GPU的算力是按照架构区分的，不是说新版本的架构一定比低版本的运算能力强，但新版本的架构性能肯定更强劲，比如算力7.5的AGX Xavier支持int8,float16,float32,Tensor\_Core等运算，但算力5.3的jetson Nano只支持float16，float32。

2.CUDA编程模型

CUDA编程模型其实很简单，Host端把需要Device计算的数据拷到显存里，由Device端的kernel完成计算后将计算结果传回Host端就OK了。

![](https://i-blog.csdnimg.cn/blog_migrate/c83a25a20c7dc39f19a5e9052a45d8cd.png)

这个过程设计到一些CUDA的基本函数，

内存分配函数：cudaMalloc

数据拷贝函数：cudaMecpy

内存释放函数：cudaFree

（省略函数对应参数）

此外，还需要了解Thread、Block、Grid的相互关系以及索引办法，这些是设计一个kernel函数调用线程的核心。Thread算是GPU硬件能够直接调用的最底层工作单元了，对应硬件的SP；多个Thread在逻辑上构成了Block，对应硬件的SM；多个Block在逻辑上构成了Grid，对应硬件的Device，就是GPU。（这种对应关系不是特别准确，因为SP，SM还包括了内存和线程控制单元，主要是为了理解）

![](https://i-blog.csdnimg.cn/blog_migrate/cf65ef32cca4fc2b02189ce47239ef17.png)

3.内存层次结构

CUDA内存种类多样，包括全局内存、共享内存、局部内存、寄存器等，内存大小依次递减，但访问速度依次递增，或者说访问时延递减，访存优化就是结合计算数据量和内存大小优化kernel，实现低时延、高吞吐量运算，个人感觉这块比较难啃。

这一块可以结合矩阵乘法取理解，搜索关键字GEMM，[GEMM解读](http://jackwish.net/2019/gemm-optimization.html "GEMM解读")，插一句，NVIDIA提供了白嫖接口可直接调用计算向量、矩阵乘法，叫CUBLAS，这个函数是列优先存储的，搞明白行列优先存储原理也就可以灵活调用这个函数了。

![](https://i-blog.csdnimg.cn/blog_migrate/cab4e8c9673e0b9b29dc611b9b04bdae.gif)

4\. Stream、Event

Stream，流，CUDA线程执行的载体， 在主机端分配设备主存（cudaMalloc），主机向设备传输数据（cudaMemcpy），核函数启动，复制数据回主机（Memcpy）以流为载体。

流有两种，隐式流和显式流，顾名思义，如果在执行kernel时没定义就是隐式的，反之就是显式的，隐式流在CUDA程序中是同步执行，显式流是异步执行，对程序速度会有影响。

Event是判断Stream是否完成的一个工具，一般用于Stream执行时间计时。

5.TensorRT

NVIDIA大杀器，深度学习的模型加速推理工具，半开半闭源状态，采用的优化技术如下所示，主要包括五项：模型量化、算子融合、核函数自动调整、动态访存、多流执行；这些也是CUDA编程的核心技术。

![](https://i-blog.csdnimg.cn/blog_migrate/d6f07224e29c4b98438f74c407fe83b4.png)

个人感觉这些技术核心就是访存优化，设计核函数减少访问数据次数，实现高性能计算。比如，量化技术，32位矩阵乘法经过量化变成16位或者8位，数据小了，一次能访问的数据个数多了，访存自然就下降了，当然主要还是Nvidia 显卡指令集（DP4A）的加持，动态访存、算子融合同理，核函数自动调整还没调研，应该也涉及访存优化技术，多流执行则是拆解数据，隐藏延迟，提高并行度来提速。

深入分析也就理解了为什么TensorRT生成的engine为什么是硬件绑定的了，因为不同硬件平台的SM数量、算力都不同，调优都是结合当下的硬件资源完成的。

目前基于TensorRT的模型优化路线主要分成几类：

（1）ONNX2TRT

（2）Torch2TRT，tensorflow2TRT

（3）TensorRT API直接从底层搭建 

三种路线的易用性、可扩展性、难度各有优缺点，大佬一般通吃，我这种菜鸡一般直接白嫖。

TensorRT支持的算子发展速度赶不上Pytorch、Tensorflow的发展速度，毕竟我写博客这功夫，又有十几篇深度学习的论文诞生了，新的算子层出不穷，需要用户自己去定义算子，Plugin应用而生，目前还没看懂怎么个用法，等看明白了再说吧。

CUDA编程涉及到的基础知识差不多就这些，每一项技术深入研究都得耗费很大功夫，但值得！挂几个链接，去看看大佬是怎么讲的：

[CUDA基础](https://mp.weixin.qq.com/s/kxYSw_fR4QMZ2-O5fvOR8g "CUDA基础")

[CUDA基础](https://zhuanlan.zhihu.com/p/97044592 "CUDA基础")

看到这儿都不点赞，我真要跳出屏幕KO你了！