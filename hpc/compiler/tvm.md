## paddle CINN
https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/cinn_cn.html

### CINN 神经网络编译器


#### 一、概念简介[](# "此标题的永久链接")


深度学习编译器是一种专门为深度学习模型优化和部署而设计的工具，用于提高模型的计算效率、降低内存占用、加速训练推理过程。其功能是将高层次的深度学习模型转换为低层次的、高效的、底层硬件可执行的代码。简单来说，深度学习编译器在深度学习框架和底层硬件之间充当了“翻译”的角色，能够将用户定义的神经网络模型描述转化为底层硬件能够理解和执行的指令。编译器在实现这种转换的过程中，应用了一系列优化技术，以提高模型在各种硬件平台上（如 CPU、GPU）的执行效率。 深度学习编译器的主要功能包括：

*   **模型转换**：将高层次的深度学习模型转换为适合目标硬件的中间表示（IR）。
    
*   **优化**：应用各种编译优化技术，如图优化、内存优化、算子融合等，以提高执行效率。
    
*   **代码生成**：生成适合目标硬件的可执行代码。
    

#### 二、背景与动机


深度学习模型的训练和推理过程涉及大量的计算，对硬件性能要求很高。飞桨框架虽然提供了高级的编程接口和丰富的算子库，但在执行效率和模型部署方面还有很大的优化空间。使用深度学习编译器的主要动机包括：

##### 1\. 优化性能与资源利用率[](# "此标题的永久链接")

深度学习模型往往需要处理大量的数据和复杂的计算，直接在高层次框架上执行可能无法充分利用底层硬件的能力。深度学习编译器能够深入硬件特性，应用多种优化技术，提高计算效率，降低延迟。并且通过优化模型的计算图和内存使用，深度学习编译器也能够明显降低模型的内存和 IO 资源的消耗，进而提高计算性能。

##### 2\. 硬件多样性支持[](# "此标题的永久链接")

不同的硬件平台有不同的特性和优化需求。在现有机制下，新的异构硬件设备接入深度学习框架需要手工实现几百个算子对应的硬件 Kernel 代码，开发的工作量非常大。如果使用深度学习编译器，理论上仅需实现新硬件 IR 层面的对接，以及相应的硬件 IR 优化策略就能完成与深度学习框架的对接，相比于实现几百个硬件 Kernel，开发的工作量会大幅减少。

##### 3\. 提升开发效率[](# "此标题的永久链接")

深度学习编译器可以自动化许多优化过程，减少手动调优的工作量。开发者只需关注模型的设计和训练，而不必深入了解底层硬件优化细节，从而提高开发效率。

#### 三、使用示例：[](# "此标题的永久链接")


飞桨框架编译器（CINN, Compiler Infrastructure for Neural Networks）使用时仅需在原先的模型动转静或推理流程下打开编译器相关 FLAGS 即可，无需对模型代码做任何改动。以下是一个使用样例：

示例代码文件：`run_net.py`

import paddle
from paddle import nn
from paddle.static import InputSpec

\# 定义神经网络
class RMSNorm(nn.Layer):
    def \_\_init\_\_(self):
        super().\_\_init\_\_()
        paddle.seed(2024)
        self.hidden\_size \= 768
        self.weight \= paddle.randn(\[self.hidden\_size\], dtype\="float32")
        self.variance\_epsilon \= 1e-6

    def forward(self, hidden\_states):
        variance \= (hidden\_states \* hidden\_states).sum(\-1, keepdim\=True) / 768
        hidden\_states \= (
            paddle.rsqrt(variance + self.variance\_epsilon) \* hidden\_states
        )
        return hidden\_states \* self.weight

def run\_net(input\_data):
    net \= RMSNorm()

    \# 指定输入变量的维度、数据类型等信息，具体接口可参考：
    \# https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/basic\_usage\_cn.html#inputspec
    input\_spec \= \[
        InputSpec(shape\=\[1, None, 768\], dtype\='float32'),
    \]
    net \= paddle.jit.to\_static(
            net,
            input\_spec\=input\_spec,
            full\_graph\=True,
        )
    \# 使用 eval 模式
    net.eval()
    \# 执行计算图
    out \= net(input\_data)
    return out

\# 创建输入数据
input\_data \= paddle.randn(\[1, 2048, 768\], dtype\="float32")
\# 运行神经网络
out \= run\_net(input\_data)
print(out)

脚本执行：`run.sh`

\# 打开组合算子
export FLAGS\_prim\_enable\_dynamic\=true && export FLAGS\_prim\_all\=true

\# 打开 CINN 编译器
export FLAGS\_use\_cinn\=true

\# 是否打印 Program IR 信息 (用于调试)
export FLAGS\_print\_ir\=false

python run\_net.py

上述代码示例中我们创建了一个简单的`rms_norm`计算子图，使用飞桨的动转静流程将子图转为静态图并调用编译器 CINN 进行优化和执行。经过性能对比测试，在 A100 GPU 环境中上述子图使用 CINN 可以取得 3 倍左右的性能提升（该性能数据仅供学习参考，在实际应用模型中能够取得的性能提升效果一般会低于该数据）。

注：由于飞桨的编译器仍然处在快速迭代开发阶段，我们设置了多个 FLAGS 进行分支的选择和调试，因此现阶段在使用 CINN 时需要对如下 FLAGS（ `FLAGS_use_cinn`、`FLAGS_prim_enable_dynamic`、`FLAGS_prim_all`） 进行手动设置，待后续相关功能完备后动转静流程将默认开启这些 FLAGS，无需再手动设置。

#### 四、设计架构[](# "此标题的永久链接")


图 1 CINN 整体架构

  

飞桨框架编译器（CINN, Compiler Infrastructure for Neural Networks）整体架构如上图所示，大体可以分为三个模块，分别是编译器前端、编译器后端和执行器部分。

##### 1\. 编译器前端[](# "此标题的永久链接")

一般来说编译器前端需要将不同框架和格式的深度学习模型转换为编译器的内部 IR 并进行图级别的优化，CINN 作为飞桨框架原生编译器，可以直接使用飞桨框架提供的模型加载和中间表示（Paddle IR，简称 PIR）组件，因此 CINN 前端的主要功能是基于 PIR 进行图层级别的优化，并对子图进行划分为后端高性能 Kernel 代码生成提供支持。CINN 前端关键的流程可分为三部分：

###### a. 组合算子拆分[](# "此标题的永久链接")

飞桨框架中将算子划分为基础算子（也称作原子算子，语义上该算子无法更进一步拆分成其他算子。基础算子语义上可以通过重组等价实现组合算子的逻辑）和非基础算子两类大，由于非基础算子数量较多，并且在编译器中较难识别和处理，因此我们使用组合算子拆分的方式将非基础算子拆分为等价的基础算子组合，原始计算图经过组合算子拆分后可以大幅提升性能的可优化空间。

###### b. 图优化 Pass[](# "此标题的永久链接")

在计算图层级进行 PIR 的 Pass 优化，常见的图优化 Pass 包括：常量折叠、死代码消除（DCE）、公共子表达式消除（CSE）、冗余算子消除、算子计算合并等。

###### c. 算子融合[](# "此标题的永久链接")

算子融合是编译器前端非常重要的一个功能，主要是将多个算子打包到一个子图中（对应为一个 FusionOp），交给编译器后端生成一个高效的硬件相关计算 Kernel。 算子融合的本质是通过 IO 优化加速访存密集算子，如果我们将两个连续 Kernel 合并为一个 Kernel 调用，我们会减少中间变量的读写开销，因此在访存密集型的 2 个 Op 上，融合可以获取更高的性能。举个例子，如下图：

![](https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/cinn/op_fusion.png?raw=true)

图 2 算子融合示例

  

我们有两个算子 Relu 和 Scale，因为两个算子都是 IO 密集型算子（计算复杂度不高）。正常情况下我们需要读取 A 和 B 一次，写 B 和 C 一次。但是对于融合之后的 Kernel（右图）而言，我们只需要读取 A 和写 C 一次，这样我们通过算子融合可以取得更少的访存次数，在 IO 密集算子而言，可以极大提高性能。 具体的算子融合策略实现非常复杂，这里不做展开介绍，感兴趣的读者可以阅读相关源码 [#cinn\_group\_cluster\_pass](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.cc)。

##### 2\. 编译器后端[](# "此标题的永久链接")

编译器后端主要负责将前端处理后的 IR 转换为目标硬件可执行的代码或硬件描述。主要功能包括基于硬件特性的 IR 优化、高效内存管理和代码生成等。

###### 2.1. CINN AST IR[](# "此标题的永久链接")

AST IR 打印示例：

ScheduleBlock(root)
{
  serial for (i, 0, 32)
  {
    serial for (j\_0, 0, 64)
    {
      serial for (j\_1, 0, 128)
      {
        ScheduleBlock(A)
        {
          vi, vj \= axis.bind(i, j\_0 \* 64 + j\_1)          // tensor 下标与循环变量的仿射变换
          A\[vi, vj\] \= X\[vi, vj\] \* 2
        }
      }
    }
  }
}

CINN AST IR 中包含了以下信息，但集合和映射并不显示使用某种数据结构进行存储。

  **集合**：语句实例 & 内存单元  **映射**：   访存关系：语句实例 <---> 内存单元   依赖关系：语句实例 <---> 语句实例   执行顺序：语句实例 -----> 语句实例

  执行顺序 = 语句实例的先后关系  语句实例集合范围 = 循环边界 + 循环步长 ------ 循环构成一个带约束的整数空间，即迭代空间，迭代空间决定了语句实例，语句实例充满了迭代空间。

###### 2.2. 基于 AST IR 的 Schedule[](# "此标题的永久链接")

Schedule 为定义在 CINN AST IR 上的优化策略，常见的 Schedule 包括：LoopAlignment, Tile, Inline, Vectorize, Unroll 等。以一个组合算子为例模拟可能的 AST 变换过程： \[S1, S2, 1024\] ==E=> \[S1, S2, 1024\] ==R=> \[S1, S2\] ==E=> \[S1, S2\] ==B=> \[S1, S2, 1024\] ==E=> \[S1, S2, 1024\]

**(1) LowerToAst 得到的结果**

// Elemenwise\-1
serial for (i, 0, S1)
  serial for (j, 0, S2)
    serial for (k, 0, 1024)
      ScheduleBlock(A)
        vi, vj, vk \= axis.bind(i, j, k)
        A\[vi, vj, vk\] \= X\[vi, vj, vk\] \* 2
// Elemenwise\-2
serial for (i, 0, S1)
  serial for (j, 0, S2)
    serial for (k, 0, 1024)
      ScheduleBlock(B)
        vi, vj, vk \= axis.bind(i, j, k)
        B\[vi, vj, vk\] \= A\[vi, vj, vk\] + 1
// Reduce\-1
serial for (i, 0, S1)
  serial for (j, 0, S2)
    ScheduleBlock(C\_\_reduce\_init)
        vi, vj \= axis.bind(i, j)
        C\_init\[vi, vj\] \= 0
serial for (i, 0, S1)
  serial for (j, 0, S2)
    serial for (k, 0, 1024)  // Reduce
      ScheduleBlock(C)
        vi, vj, vk \= axis.bind(i, j, k)
        C\[vi, vj\] \= C\[vi, vj\] + B\[vi, vj, vk\]
// Elemenwise\-3
serial for (i, 0, S1)
  serial for (j, 0, S2)
    ScheduleBlock(D)
      vi, vj \= axis.bind(i, j)
      D\[vi, vj\] \= C\[vi, vj\] \* 2
// Broadcast\-1
serial for (i, 0, S1)
  serial for (j, 0, S2)
    serial for (k, 0, 1024)  // Broadcast
      ScheduleBlock(E)
        vi, vj, vk \= axis.bind(i, j, k)
        E\[vi, vj, vk\] \= D\[vi, vj\]
// Elemenwise\-4
serial for (i, 0, S1)
  serial for (j, 0, S2)
    serial for (k, 0, 1024)
      ScheduleBlock(F)
        vi, vj, vk \= axis.bind(i, j, k)
        F\[vi, vj, vk\] \= E\[vi, vj, vk\] + 1

**(2) 迭代空间对齐**

// 所有 ScheduleBlock 的 loop nest 都变为以下 2 种格式中的一种
// 1
serial for (sp, 0, S1 \* S2)  // pure\_spatial\_iter
  serial for (rb, 0, 1024)    // impure\_spatial\_iter
    ScheduleBlock(XXX)
      vsp1, vsp2, vrb \= axis.bind(sp / S2, sp % S2, rb)
      XXX \= XXXXXX
// 2
serial for (sp, 0, S1 \* S2)  // pure\_spatial\_iter
   ScheduleBlock(XXX)
     vsp1, vsp2 \= axis.bind(sp / S2, sp % S2)
     XXX \= XXXXXX

**(3) Tile: 对所有 ScheduleBlock 的 loop nest 做相同的 Tile**

// pure\_spatial 轴 Tile 为：-1 \* 16 \* 64   Tile size 可为参数传入
serial for (sp1, 0, S1 \* S2 / 1024)
  serial for (sp2, 0, 16)
    serial for (sp3, 0, 64)     // S1 \* S2 / 16 / 64, predicate: sp1 \* 1024 + sp2 \* 16 + sp3 < S1 \* S2
      XXXXXX
// impure\_spatial\_iter 轴 Tile 为 32
serial for (sp1, 0, S1 \* S2 / 1024)
  serial for (sp2, 0, 16)
    serial for (sp3, 0, 64)
      serial for (rb1, 0, 32)
        serial for (rb2, 0, 32)
          ScheduleBlock(XXX)
            predicate \= sp1 \* 1024 + sp2 \* 16 + sp3 < S1 \* S2
            vsp1 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
            vsp2 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
            vrb = axis.bind(rb1 \* 32 + rb2)
            XXX = XXXXX

**(4) ComputeInline**

// 例如 ScheduleBlock(A) inline 到 ScheduleBlock(B)
serial for (sp1, 0, S1 \* S2 / 1024)
  serial for (sp2, 0, 16)
    serial for (sp3, 0, 64)
      serial for (rb1, 0, 32)
        serial for (rb2, 0, 32)
          ScheduleBlock(A)
            predicate \= sp1 \* 1024 + sp2 \* 16 + sp3 < S1 \* S2
            vsp1 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
            vsp2 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
            vrb \= axis.bind(rb1 \* 32 + rb2)
            B\[vsp1, vsp2, vrb\] \= (X\[vsp1, vsp2, vrb\] \* 2) + 1

**(5) Reduce 优化: two step reduce & 绑定部分 reduce 轴到 cuda**

// 为了简洁，此处省略 reduce\_init Block 和 predicate
serial for (sp1, 0, S1 \* S2 / 1024)
  serial for (sp2, 0, 16)
    serial for (sp3, 0, 64)
      CudaBind\[ThreadIdx.x\] for (rb1, 0, 32)
        serial for (rb2, 0, 32)
          ScheduleBlock(C\_rf)
            vsp1 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
            vsp2 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
            vrb1 = axis.bind(rb1)
            vrb2 = axis.bind(rb2)
            C\_rf\[vsp1, vsp2, vrb1\] = C\_rf\[vsp1, vsp2, vrb1\] + B\[vsp1, vsp2, vrb1 \* 32 + vrb2\]
        ScheduleBlock(C)
          vsp1 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
          vsp2 = axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
          vrb1 = axis.bind(rb1)
          C\[vsp1, vsp2\] = C\[vsp1, vsp2\] + C\_rf\[vsp1, vsp2, vrb1\]

**(6) 循环融合: ComputeAt && SimpleComputeAt，融合外层循环乘积相同的循环，并且保证不破坏图级别依赖（规则负责）和元素级别依赖（原语负责）**

serial for (sp1, 0, S1 \* S2 / 1024)
  serial for (sp2, 0, 16)
    serial for (sp3, 0, 64)
      ScheduleBlock(D)
        vsp1 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
        vsp2 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
        D\[vsp1, vsp2\] \= C\[vsp1, vsp2\] \* 2
      serial for (rb1, 0, 32)
        serial for (rb2, 0, 32)
          ScheduleBlock(E)
            vsp1 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
            vsp2 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
            vrb \= axis.bind(rb1 \* 32 + rb2)
            E\[vsp1, vsp2, vrb\] \= D\[vsp1, vsp2\]
          ScheduleBlock(F)
            vsp1 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) / S2)
            vsp2 \= axis.bind((sp1 \* 1024 + sp2 \* 16 + sp3) % S2)
            vrb \= axis.bind(rb1 \* 32 + rb2)
            F\[vsp1, vsp2, vrb\] \= E\[vsp1, vsp2, vrb\] + 1

**(7) Bind Cuda 轴：在第二步中，所有 ScheduleBlock 对应的循环要 bind 到同一 Cuda 轴**

serial for (sp1, 0, S1 \* S2 / 1024)
  CudaBind\[BlockIdx.x\] for (sp2, 0, 16)
    CudaBind\[ThreadIdx.y\] for (sp3, 0, 64)
      CudaBind\[ThreadIdx.x\] for (rb1, 0, 32)
        serial for (rb2, 0, 32)
          ScheduleBlock(XXX)

###### 2.3. Kernel 代码生成与编译[](# "此标题的永久链接")

Codegen 在 CINN IR AST 上做前序遍历，打印出对应硬件的指令，并通过硬件相对应的编译器（如 llvm、nvcc 等）进行编译得到可运行的函数指针，该指针会被封装到 \`JitKernelOp\`\` 中用于后续执行器的解析执行。

a. 以函数定义为例子，cuda kernel func 和 x86 kernel func 的不同的是，cuda kernel func 会在函数名前增加 `__global__`

针对 x86 硬件，转义 `ir::_LoweredFunc_` 的代码如下：

void CodeGenC::Visit(const ir::\_LoweredFunc\_ \*op) {
  PrintFunctionDeclaration(op); // 前序遍历继续转义函数名、函数参数等
  str\_ += "\\n";
  ...
  ...
}

在 NV GPU 上的转义代码如下：

void CodeGenCUDA\_Dev::Visit(const ir::\_LoweredFunc\_ \*op) {
  str\_ += "\_\_global\_\_\\n";       // 和 x86 的不同，增加 \_\_global\_\_
  PrintFunctionDeclaration(op); // 前序遍历继续转义函数名、函数参数等
  str\_ += "\\n";
  ...
  ...
}

b. 在动态形状场景下，还会 codegen 出 infer shape function, infer shape function 的 CINN IR 会在 Bucket Lowering 中得到，转义过程复用的 x86 硬件的 codegen。infer shape kernel 如下：

// infer shape 函数名字的组成：kernel\_name + "infer\_shape"
// 函数参数：
//     kernel\_args: 指针数组，和 kernel func args 一致
//     kernel\_args\_num: kernel\_args 的长度
//     tensor\_shape\_args: 指针数组，存储输出 tensor 的 shape
function fn\_exp\_0\_subtract\_0\_infer\_shape (kernel\_args, kernel\_args\_num, tensor\_shape\_args)
{
  int64 S0 = cinn\_get\_value\_in\_cuda\_kernel\_args(kernel\_args, 2)
  {
    // CINN IR 暂时不支持数据索引的语法，暂时用函数调用实现，下面 2 条语句等价于
    //   tensor\_shape\_args\[0\] = {S0, 256ll};
    // 即第 0 个出 tensor 的 shape 为{S0, 256ll};
    infer\_shape\_set\_value(0, 0, S0, tensor\_shape\_args)
    infer\_shape\_set\_value(0, 1, 256ll, tensor\_shape\_args)
  }
}

##### 3\. 执行器[](# "此标题的永久链接")

编译器生成的 Kernel 代码需要与深度学习框架执行器完成交互和集成才能最终运行起来，因此需要基于执行器的运行调度接口对编译器生成的 Kernel 进行封装。

接入执行器后在运行时对于经过编译器处理的子图将执行 CINN 生成的 Kernel, 否则将执行常规的 PHI 算子 Kernel。

#### 五、技术创新点[](# "此标题的永久链接")


1.  以 Reduce 为核心的算子深度融合
    
    *   摒弃传统的粗粒度 pattern 匹配模式，支持维度轴自动变换对齐融合，在保证计算正确性的同时，具有更强的算子融合能力，带来更大的性能优化潜力。
        
2.  动静态维度的高效后端 Kernel 调优
    
    *   算子支持全面：支持 reduce、broadcast、transpose 等多种算子的不同组合方式，针对各类算子组合和数据类型，自适应不同维度大小与不同硬件配置，进行全场景高效调优。
        
    *   自动向量化加速：提高 BF16、FP16 等小数据类型的访存效率。
        
    *   动静态运行时配置生成：通过分析与分桶机制，根据运行时的硬件配置，在无需 profiling 的情况下生成高效 Kernel。
        
3.  动态维度的复杂表达式化简
    
    *   分层化简体系：Lower、Schedule、CodeGen 阶段执行不同等级化简方法，解决传统化简方法中多场景叠加后化简困难、化简不彻底问题。
        
    *   复杂表达式结构化简：深度解析融合算子编译调优后的固定子结构，建立专项化简模板，同时开放自定义化简接口，实现化简策略的弹性扩展与精准适配。
        

#### 六、性能优化效果[](# "此标题的永久链接")


1.  科学计算系列模型：
    
    *   飞桨基于 CINN 编译器在科学计算 Modulus 模型上，相比 Pytorch 求解速度平均提升 115%。
        

![](https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/cinn/modulus_cinn_vs_torch.png?raw=true)

图 3 Modulus 系列模型性能对比数据

  

1.  PaddleX 模型：
    
    *   PaddleX 系列 60+ 模型使用 CINN 编译器后超 60% 模型有显著性能提升，平均提升达 27.4%。部分重点模型相比 Pytorch 也有明显性能优势。
        

![](https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/cinn/PaddleX_cinn_vs_torch.png?raw=true)

图 4 部分重点模型单机 8 卡 FP16 训练性能对比数据


## pytorch中的动态图和静态图

`torch.compile` 和 `torch.jit` 都是 PyTorch 中用于优化模型执行的工具，但它们的**设计目标、工作方式和适用场景**有本质区别。简单来说：`torch.jit` 核心是“动静转换与部署”，`torch.compile` 核心是“动态图性能加速”。


### 一、设计目标：解决的问题不同
- **`torch.jit`**：  
  诞生于 PyTorch 早期（1.0+），核心目标是**实现“动态图开发，静态图部署”**。  
  它希望解决动态图的两个痛点：1）动态图依赖 Python 解释器，难以在无 Python 环境的生产端（如 C++ 服务器、嵌入式设备）部署；2）动态图的即时执行模式存在额外开销，不利于推理性能优化。  
  因此，`torch.jit` 的核心是将灵活的动态图代码转换为“可序列化的静态图”，方便部署和基础优化。

- **`torch.compile`**：  
  诞生于 PyTorch 2.0（2023 年），核心目标是**在保持动态图易用性的前提下，大幅提升模型执行性能**（尤其是训练和推理速度）。  
  它针对 PyTorch 动态图的性能瓶颈（如 Python 解释器开销、算子调度效率低），通过更先进的编译技术对动态图进行深度优化，同时不牺牲动态图的灵活性（如支持 Python 原生控制流、实时调试）。  


### 二、工作方式：从输入到输出的逻辑不同
#### 1. `torch.jit` 的工作逻辑：“静态转换”
`torch.jit` 本质是“将动态图代码转换为静态图表示”，主要通过两种方式：  
- **`torch.jit.trace`**：执行一次模型，记录算子调用轨迹（Trace），生成静态图（忽略控制流细节）；  
- **`torch.jit.script`**：解析 Python 代码的 AST（抽象语法树），将动态图代码转换为 TorchScript 静态 IR（中间表示），支持有限的控制流。  

转换后的静态图是**“固定结构”**，可以被序列化（保存为 `.pt` 文件），且脱离 Python 解释器执行（如通过 C++ API 加载）。但其优化能力较基础（如简单的算子融合、常量折叠），且对 Python 动态语法（如 `exec`、动态属性访问）支持有限。


#### 2. `torch.compile` 的工作逻辑：“动态编译优化”
`torch.compile` 本质是“对动态图进行即时编译（JIT）并生成优化代码”，核心流程是：  
- 输入用户编写的**动态图模型**（无需修改代码，支持所有 Python 语法）；  
- 内部将动态图转换为更灵活的 **TorchIR**（PyTorch 2.0 自研中间表示，比 TorchScript IR 更贴近动态图特性）；  
- 通过**编译器优化 passes**（如深度算子融合、内存复用、控制流简化、硬件感知布局调整）对 TorchIR 进行优化；  
- 最终通过 **TorchInductor**（默认后端）生成目标硬件的高效代码（如 CPU 的优化 C++ 代码、GPU 的优化 CUDA 代码）。  

`torch.compile` 不改变动态图的“开发体验”——用户仍可像写普通动态图一样调试、修改模型，但执行时会自动使用编译后的优化代码，性能大幅提升。


### 三、核心差异对比
| 维度                | `torch.jit`                              | `torch.compile`                          |
|---------------------|------------------------------------------|------------------------------------------|
| **核心目标**        | 动静转换，支持部署（脱离 Python）        | 性能加速（训练/推理），保持动态图灵活性  |
| **对动态语法支持**  | 有限（`script` 对复杂语法转写易失败）    | 几乎完全支持（基于 TorchIR 动态追踪）    |
| **优化能力**        | 基础（简单算子融合、常量折叠）           | 深度优化（复杂融合、内存优化、硬件适配）  |
| **输出形式**        | 序列化静态图（`.pt`），可被 C++ 加载     | 编译后的优化模型（`CompiledModule`），依赖 Python 执行环境（目前） |
| **适用场景**        | 生产环境部署（如移动端、C++ 服务器）     | 研究/生产中的训练加速、Python 环境下的推理加速 |
| **用户干预**        | 需要适配静态语法（如避免动态控制流）     | 无需修改代码，直接包裹模型即可           |


### 四、举例说明：实际使用的区别
#### 1. `torch.jit` 的典型用法（部署导向）
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + 1

# 转换为静态图
scripted_model = torch.jit.script(MyModel())
# 序列化保存（部署用）
torch.jit.save(scripted_model, "model.pt")
# 后续可通过 C++ API 加载执行（无需 Python）
```


#### 2. `torch.compile` 的典型用法（性能导向）
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # 支持复杂动态逻辑（如依赖输入的控制流）
        if x.sum() > 0:
            return x * 2
        else:
            return x / 2

# 直接编译优化（无需修改模型代码）
compiled_model = torch.compile(MyModel())

# 执行时自动使用优化代码（速度更快）
x = torch.randn(1000)
print(compiled_model(x))  # 保持动态图的灵活性和调试体验
```


### 五、总结：如何选择？
- 若需要**将模型部署到无 Python 环境的生产端**（如 C++ 服务器、嵌入式设备），用 `torch.jit`；  
- 若需要**提升动态图模型的训练/推理速度**（且模型运行在 Python 环境中），用 `torch.compile`；  
- 两者并非替代关系：可以先用 `torch.compile` 加速训练，再用 `torch.jit` 将训练好的模型转换为静态图部署。  

`torch.compile` 代表了 PyTorch 向“动态图易用性 + 静态图高性能”融合的方向，而 `torch.jit` 仍是部署场景的核心工具。