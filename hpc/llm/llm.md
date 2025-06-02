_在之前的文章中，我们已经了解了Transformer的基本架构和Attention的机制思想：_

  

[yzzz：前言---Transformer与LLM的概念解释与理解18 赞同 · 3 评论文章](https://zhuanlan.zhihu.com/p/4332271801)

  

[yzzz：理解Attention---从人类思考过程理解Attention15 赞同 · 0 评论文章](https://zhuanlan.zhihu.com/p/4478989317)

  

而本文的目的是为了探究LLM在推理过程中[Attention算子](https://zhida.zhihu.com/search?content_id=249939627&content_type=Article&match_order=1&q=Attention%E7%AE%97%E5%AD%90&zhida_source=entity)的真实细节。事实上由于现在网上大部分人们在介绍LLM的Attention或者LLM时，都会或多或少简化其中的计算过程。并且在描述流程时，对模型的参数名称进行了各式各样的修改。这就导致在我们想要尝试自己计算一个LLM模型大小参数、或者一次前向推理需要多少次计算操作时，会发现根本无从下手。_

_故本文将会通过具体的矩阵计算过程图来探究在Attention机制究竟是如何在大语言模型LLM的推理过程中发挥作用的。_

* * *

  

在现在大部分LLM当作，多采用Multi-SelfAttention多头的注意力机制。但如果在不清楚Attention计算流程的情况下直接了解多头注意力会感觉较为复杂。所以本文将先介绍单头的SelfAttention的计算过程。（注意！[Self-Attention](https://zhida.zhihu.com/search?content_id=249939627&content_type=Article&match_order=1&q=Self-Attention&zhida_source=entity)的计算过程只对理解Attention机制有帮助，其实际的参数和计算过程与多头注意力机制有较大的出入）

### 一、Self-Attention

话不多说，我们直接看图：

(Self-Attention的流程图主要用于理解Attention计算过程。更严格地说，只表达了LLM中Perfill的过程，而Decode的流程将会在后续给出。

![](https://pic1.zhimg.com/v2-d37ce841468046d14a2c18cc74fa599c_r.jpg)

下面进行一些变量名称解释与约定：

1.  流程图中矩阵块的长和宽严格遵守了矩阵乘法的规则
2.  名称为**黄色**的都是**参数矩阵**，名称为**紫色**的是输入**流动的数据**。
3.  S 为输入文本的序列长度，例如输入：“同学们大家好”，那么S应该为 6。
4.  V为预训练时词汇表的长度，通常大小为250,000左右。
5.  d\_model： 模型隐变量维度，是衡量模型参数大小的重要指标。也是词嵌入层的输出维度。
6.  Matmul：矩阵乘操作。
7.  Scale ： 缩放操作，就是原公式中那个除以 根号dk 的操作

可以看到，如果仅仅是单头的自注意力机制，那么在整个Attention过程中，计算**参数**只有词嵌入矩阵和词嵌入位置矩阵。（输出矩阵 WOW\_OW\_O 与词嵌入矩阵共享参数，即转置关系）。

我们会发现整个注意力相关的计算过程没有什么可以学的参数。所以作者借鉴了CNN网络计算过程中“多通道”的思想：即对一个单通道的样本拓展至多通道数，来增加学习到特征的多样性。从而就有了多头注意力机制。

接下来让我们一起看一看多头注意力机制的计算过程：

* * *

  

### 二、Multi-SelfAttention

下面的计算计算过程是严格按照Transformer论文中数据的计算格式和名称进行绘制的。在后续计算LLM的推理计算量和模型参数时，可直接使用该图作为参照标准。

![](https://picx.zhimg.com/v2-39914384b022c7265f2ce8130370e597_r.jpg)

（注意：变量h为多头注意力机制的头数。改图中h应该为2，虽然图中画了3层，那只是为了说明该矩阵是一个三维矩阵，并不代表其高h为3。并且作者规定：d\_model = d\_k \* h，后续会讲解为什么）

我们可以看到，整个流程被分为了几块：[Vocabulary Embedding](https://zhida.zhihu.com/search?content_id=249939627&content_type=Article&match_order=1&q=Vocabulary+Embedding&zhida_source=entity)词嵌入层、[Linear线性映射层](https://zhida.zhihu.com/search?content_id=249939627&content_type=Article&match_order=1&q=Linear%E7%BA%BF%E6%80%A7%E6%98%A0%E5%B0%84%E5%B1%82&zhida_source=entity)、[Scale-Dot-Product Attention](https://zhida.zhihu.com/search?content_id=249939627&content_type=Article&match_order=1&q=Scale-Dot-Product+Attention&zhida_source=entity) 缩放点乘注意力层。

其中Scale-Dot-Product Attention层的计算过程与之前的SelfAttention中的计算过程完全对等，只不过将原本的二维的Q、K、V向量矩阵变成了三维张量Tensor。但是在具体计算过程中，还是每个二维的向量切片做计算，最后叠加至三维。

此过程与SelfAttention最大区别是，多了一个Linear线性映射层。那么怎样通俗的理解这一层在做什么呢？

让我们举一个具体的例子：假设现在由词嵌入矩阵输出了一个 5\\times405\\times40 的长方形矩阵，并且将其当作Q、K、V。那么现在定义一个映射矩阵W，其大小为 40\\times1040\\times10 。此时 Q\\cdot WQ\\cdot W 的输出是一个 5\\times105\\times10 的矩阵。

在这个过程中，W起到了什么作用？没错，就是降低维度，将一个长40维的向量线性映射到10维。那么如果我使用4个相同形状的W矩阵分别对Q进行映射操作，就会得到4个相同的被映射至低维的向量。

在进行QKV的点积关联计算之后，我们就得到了4个5\\times105\\times10 的平面二维矩阵。然后将二维矩阵竖着叠加起来，也就变成了一个 5\\times10\\times45\\times10\\times4 的三维向量。这就是多头注意力机制多通道的过程，你使用了多少个W进行低维映射，就等价于多头注意力有几个头。

只不过这里有一个约定：在多头注意力机制中，映射到低维的向量数量，也就是多头的头数h 与映射后的维度 d\_kd\_k 的乘积应该等于原始的维度 d d \_model。通俗的理解就是：我将一个长方形切为h份，并且把切完的每一份都竖着垒起来，变成了立方体。只有这样，在进行计算之后，才可以将三维立方体摊开，还原会原来现状的长方形（输入）

我们可以看到，在计算的最后的**Concat**拼接操作，就是这个堆砌过程的逆过程。即把一个立方体在垂直方向切成二维切片，然后拼成一个长方体。

最后，我们会发现，多头注意力机制相比普通的自主注意力机制，多了三个可学习的参数矩阵 W^Q、W^K、W^VW^Q、W^K、W^V 。这就是多头注意力机制的最终目的，使得我们的注意力相关过程可以学习、并且能够学习到不同特征。

需要注意：从图中可以得知，整个过程可学习的参数除了Embeddig的两个嵌入参数矩阵和三个多头映射矩阵W之外，还多了一个输出线性映射矩阵 W^OW^O 。它与上一节中词嵌入的输出矩阵 W^OW^O 不是一个东西，他的作用是进行等维的线性映射。实际在整个LLM推理之后，将输出转化为词嵌入的句子操作在本过程中并没有画出（篇幅源原因）。

* * *

以上便是LLM在Perfill过程中Attention算子计算的全部过程。后续推理的Decode过程、KV cache以及模型参数的计算将会后续章节中给出。