## gk版本特性
### 适配难点
1. 对于模型输入输出，在llama.cpp进行data_type和format转换
2. 对于中间结果，若牵扯到输入输出data_type转换， 需要手动插入cast算子
3. 所有的const数据，调用 gkcl_malloc 申请，数据排布的转换，在运行之前转换完毕。 
4. 在运行的过程中，const数据内部不拷贝
5. 调用gkcl_get_tensor_pitch接口 获取对应tensor的pitch信息
6. 网络输入nchw   中间层 nc1hwc0/bmn 


### 结果验证
这里的“对比CPU”指的是**对比“CPU原生推理结果”与“通过cmodel模拟NPU推理的结果”**，核心目的是验证cmodel的计算逻辑是否正确，以及模型在NPU上的执行逻辑是否与预期一致。尽管cmodel确实运行在CPU上，但二者的计算路径和意义完全不同，具体可从以下角度理解：


#### 1. 为什么cmodel运行在CPU上，却要对比CPU推理结果？
cmodel的本质是**“用CPU代码模拟NPU硬件的计算行为”**，而“CPU推理”指的是**“直接用CPU原生算子（如PyTorch、TensorFlow的CPU后端）执行模型推理”**。二者的核心区别在于：
- **计算逻辑的“目标对象”不同**：
  - cmodel的代码逻辑严格遵循NPU硬件的设计规范（如支持的算子精度、数据排布格式、计算单元的运算规则），其输出结果是“模拟NPU硬件会产生的结果”。
  - CPU推理的代码逻辑遵循通用计算规范（如IEEE浮点标准、通用内存访问方式），其输出结果是“理论上正确的参考结果”（或称为“ground truth”）。
- **对比的目的是验证“模拟的准确性”**：  
  硬件团队需要通过对比确认：cmodel是否准确复现了NPU的设计逻辑（而不是单纯验证CPU计算是否正确）。例如，若NPU的`matmul`算子因硬件限制采用了低精度截断（如FP16截断为特定格式），cmodel必须模拟这种截断行为，此时对比CPU的FP32结果就能验证截断逻辑是否符合设计预期。


#### 2. 具体对比场景：以ResNet推理为例
假设用ResNet对一张图片进行分类，对比流程如下：
- **CPU推理**：用PyTorch的`torch.nn.Conv2d`、`torch.matmul`等CPU算子执行推理，得到输出概率分布（如“猫：99%，狗：1%”）——这是基于通用计算逻辑的“参考结果”。
- **cmodel推理**：将ResNet的算子（卷积、矩阵乘法等）映射到cmodel提供的模拟接口（如`npu_conv2d`、`npu_matmul`），用CPU运行cmodel代码，得到模拟NPU执行的输出概率分布。
- **对比核心**：  
  两者的结果允许存在**合理误差**（如NPU采用低精度计算导致的精度损失），但必须在误差范围内一致（如分类结果相同，概率值差异在1e-3以内）。若差异过大，则说明：
  - 要么cmodel的模拟逻辑有误（如算子实现与硬件设计不符）；
  - 要么模型映射到NPU的过程有误（如数据格式转换错误、算子参数传递错误）。


#### 3. 延伸：为什么不直接对比cmodel与硬件接口的结果？
因为在cmodel阶段，硬件可能还未量产（甚至还在流片阶段），无法获取真实硬件的结果。因此，**CPU推理结果是唯一可立即获取的“基准参考”**。  
当硬件交付后，最终会用“硬件接口推理结果”替代“CPU推理结果”，再次与cmodel结果对比，验证cmodel的模拟是否准确反映了硬件的真实行为（这一步是硬件验收的关键环节）。


#### 总结
“对比CPU推理结果”的核心是：  
用通用计算逻辑的“已知正确结果”，验证cmodel对NPU硬件计算行为的“模拟准确性”，同时确认推理框架将模型映射到NPU的逻辑是否正确。尽管cmodel运行在CPU上，但其模拟的是NPU的计算逻辑，与CPU原生推理的计算路径完全不同，因此对比具有明确的验证意义。

### 数据排布特性
![alt text](images/image.png)
![alt text](images/image-2.png)
![alt text](images/image-1.png)
硬件排列格式
#### nc1whc0
##### 理解 NCHW 到 NC₀HWC₁ 的排布转换

`NC₀HWC₁` 是一种用于优化硬件计算的内存排布方式，常见于昇腾 NPU 等设备中。这种排布通过将通道维度（C）拆分为两个维度（C₀ 和 C₁），更好地匹配硬件的计算单元结构（如脉动阵列）。下面我将通过可视化和具体步骤解释从标准的 NCHW 到 NC₀HWC₁ 的转换过程。


##### 1. 原始 NCHW 排布

假设我们有一个形状为 `[N=1, C=4, H=2, W=2]` 的张量，其逻辑结构如下：

```
# NCHW 逻辑排布（C=4 个通道，每个通道 2×2 像素）
通道0: [[0, 1],    通道1: [[4, 5],    通道2: [[8, 9],    通道3: [[12, 13],
        [2, 3]]            [6, 7]]            [10, 11]]           [14, 15]]

# 内存中的一维排列（行优先）
[0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
```


##### 2. 确定 C₀ 和 C₁ 的值

在 NC₀HWC₁ 排布中，需要将原始通道数（C）拆分为：
- **C₀**：向量单元宽度，通常与硬件计算单元匹配（如 16 或 32）
- **C₁**：通道分组数，满足 `C = C₁ × C₀`

对于我们的例子（C=4），假设选择 `C₀=2`，则 `C₁ = C/C₀ = 4/2 = 2`。


##### 3. 重新组织数据：NCHW → NC₀HWC₁

转换步骤如下：

1. **按 C₀ 分组通道**：将 4 个通道分为 2 组（C₁=2），每组 2 个通道（C₀=2）
2. **按 HW 遍历每个位置**：对于每个空间位置 (h,w)，依次访问每组通道
3. **内存排列顺序**：`N → C₁ → H → W → C₀`

转换后的逻辑结构：

```
# 分组1（C₁=0，包含通道0和通道1）
通道0: [[0, 1],    通道1: [[4, 5],
        [2, 3]]            [6, 7]]

# 分组2（C₁=1，包含通道2和通道3）
通道2: [[8, 9],    通道3: [[12, 13],
        [10, 11]]           [14, 15]]

# 内存中的一维排列（按 NC₀HWC₁ 顺序）
[0,4, 1,5, 2,6, 3,7,    # 分组1（C₁=0）的所有位置
 8,12, 9,13, 10,14, 11,15]  # 分组2（C₁=1）的所有位置
```



##### 4. 代码示例（模拟转换过程）

下面的代码展示了如何手动实现 NCHW 到 NC₀HWC₁ 的转换：

```python
import torch

# 创建 NCHW 张量
n, c, h, w = 1, 4, 2, 2
tensor_nchw = torch.arange(n*c*h*w).reshape(n, c, h, w)

# 定义 C₀ 和 C₁
c0 = 2  # 向量单元宽度
c1 = c // c0  # 通道分组数

# 重新排列为 NC₀HWC₁
tensor_nc0hwc1 = tensor_nchw.reshape(n, c1, c0, h, w).permute(0, 1, 3, 4, 2)

print("NCHW 形状:", tensor_nchw.shape)  # 输出: [1, 4, 2, 2]
print("NCHW 内存布局:", tensor_nchw.flatten())
# 输出: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

print("NC₀HWC₁ 形状:", tensor_nc0hwc1.shape)  # 输出: [1, 2, 2, 2, 2]
print("NC₀HWC₁ 内存布局:", tensor_nc0hwc1.flatten())
# 输出: [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
```

#### 5. 为什么要这样转换？

这种排布优化主要针对硬件计算：
- **数据局部性**：同一计算单元（如脉动阵列）可高效处理连续的 C₀ 个通道数据
- **并行计算**：多个 C₁ 分组可分配到不同计算核心并行处理
- **内存访问**：减少跨缓存行的数据读取，提升吞吐量

例如，昇腾 NPU 的 AICore 采用 16×16 的脉动阵列，适合处理 C₀=16 的数据块，此时将大通道数按 C₁ 分组可充分利用硬件并行性。
#### 注意
- 修改内存访问顺序：通过调整 stride 元信息改变维度访问顺序，通常不立即拷贝数据。
- 非连续张量：生成的张量可能是非连续的（元素在内存中不按逻辑顺序排列），需调用 contiguous() 强制连续。


#### bmn
nchw -》 bm1
![alt text](images/image-3.png)
![alt text](images/image-4.png)
![alt text](images/image-5.png)
中间结果支持硬件cast函数（子数normalize）
![alt text](images/image-6.png)

fp16
fp16 + 量化

### cv
工具链编译，扔给驱动直接出结果，不需要运行时接口

### 高性能
llama.cpp传整图，编译后走完整runtime

### 具体算子实现
#### matmul
输出+reshape+transpose
llama.cpp差异（对比pytorch）
#### reshape
逻辑偏移，后序是否连续

#### sofxmax
llama.cpp里实现，softmax掩码实现在了softmax里

### 开发形态
#### docker

#### fpga

### 生图框架
#### comfy ui
#### stable diffusion.cpp
每次forward 都需要重新build，图动态转静态


## 芯片架构学习资源
- zomi博客
https://developer.aliyun.com/article/1643218?spm=a2c6h.13262185.profile.10.219e76b4yOkWuk

### SIMD 与SIMT芯片架构
- https://developer.aliyun.com/article/1643218?spm=a2c6h.13262185.profile.10.219e76b4yOkWuk
- https://developer.aliyun.com/article/1643220?spm=a2c6h.13262185.profile.9.5a0376b4ihH2o3
- https://developer.aliyun.com/article/1643242?spm=a2c6h.13262185.profile.7.5a0376b4ihH2o3

### 算子优化指标与编程范式
https://developer.aliyun.com/article/1644046?spm=a2c6h.13262185.profile.33.2c4521f77FIgCm

### gpu架构-tensorcore
https://developer.aliyun.com/article/1642224?spm=a2c6h.13262185.profile.27.219e76b4yOkWuk

### ai芯片架构能耗问题
https://www.cnblogs.com/wujianming-110117/p/18234286




## 编译器
在技术语境中，“Phoenix” 常与特定硬件或软件生态相关联。在华为昇腾（Ascend）NPU 生态中，Phoenix 是针对昇腾 NPU 的专用编译优化工具，主要用于算子的编译与优化。

它能识别昇腾 NPU 的底层原语（prim）和向量内联函数（vector intrinsic），将其转换为 NPU 硬件可执行的指令，并结合硬件特性（如并行计算单元、存储结构）进行优化，例如指令调度、数据布局调整等，以提升算子在昇腾 NPU 上的执行效率，是昇腾软件栈中连接算子代码与硬件指令的关键工具。
Phoenix 是否属于这类 NPU 专用编译器，需要结合具体语境来判断，因为“Phoenix”在不同技术场景下可能指代不同工具：

它的核心功能包括：  
- 处理基于昇腾 NPU 原语（如 `prim` 基础操作）和向量内联函数（`vector intrinsic`）的算子代码；  
- 将这些代码编译为昇腾 NPU 硬件能直接执行的指令（如适配达芬奇架构的指令集）；  
- 结合硬件特性进行优化（如指令打包、数据布局调整、并行调度等），最大化算子执行效率。  

### 1. 毕昇编译器（Bisheng Compiler）：全栈式异构编译引擎
- **核心定位**：面向昇腾全栈软硬件的**通用异构编译器**，不仅支持NPU，还覆盖CPU、GPU等多硬件，是连接上层代码与底层硬件的“全链路翻译官”。  
- **核心功能**：  
  - **多语言支持**：处理C/C++、Fortran等通用语言，以及昇腾定制的Ascend C算子开发语言（用于编写高性能NPU算子）。  
  - **双端/多端编译**：支持“一套代码编译为多硬件实现”（如同时生成Host端CPU代码和Device端NPU代码），这也是昇腾“双端编译”的核心工具。  
  - **全链路优化**：从高级语言解析、中间代码优化（如循环展开、内存布局调整）到硬件指令生成（适配昇腾NPU的达芬奇架构指令集），提供端到端优化。  
  - **生态兼容性**：兼容LLVM、GCC等通用编译框架，同时集成昇腾硬件专属的优化策略（如NPU的指令级并行调度、数据搬移优化）。  
- **典型场景**：  
  - 开发者用Ascend C编写算子代码，通过毕昇编译器同时生成可在Host（CPU）和Device（NPU）运行的二进制文件。  
  - 对深度学习框架（如MindSpore）的模型代码进行编译，适配昇腾NPU的执行逻辑。  


### 2. Phoenix 编译框架：NPU算子专用优化器
- **核心定位**：聚焦于昇腾NPU的**算子级编译优化框架**，是毕昇编译器在NPU算子编译环节的“专项优化模块”。  
- **核心功能**：  
  - **算子指令生成**：针对昇腾NPU的原语（prim）和向量内联函数（vector intrinsic），将其精准映射为硬件指令（如达芬奇架构的计算指令、数据搬移指令）。  
  - **硬件特性深度优化**：专门优化NPU的硬件特性，如指令打包（VLIW架构的超长指令字组合）、计算单元调度（充分利用NPU的MAC阵列）、片上内存（UB、L2缓存）的高效利用。  
  - **与算子库协同**：与昇腾算子库（如aicore算子库）深度绑定，为复杂算子（如卷积、矩阵乘）提供底层指令级优化。  
- **典型场景**：  
  - 当毕昇编译器将Ascend C算子代码转换为中间表示（IR）后，Phoenix负责对IR中与NPU硬件相关的部分进行精细化优化，最终生成高效的NPU机器码。  
  - 针对算子的并行性、数据局部性等进行专项调优，最大化NPU的算力利用率。  


### 总结：两者的关系与区别
| **维度**       | 毕昇编译器（Bisheng Compiler）                | Phoenix 编译框架                          |
|----------------|----------------------------------------------|------------------------------------------|
| **范围**       | 全栈异构编译（支持CPU、NPU等多硬件）          | 仅针对NPU的算子级优化                    |
| **角色**       | 端到端的“大管家”，负责从代码到多硬件二进制的全流程 | NPU算子编译的“精算师”，专注硬件指令级优化 |
| **核心目标**   | 实现多硬件兼容、双端编译，降低开发门槛        | 最大化NPU算子性能，挖掘硬件极限能力      |
| **协作方式**   | 毕昇编译器调用Phoenix作为NPU算子编译的后端模块 | 作为毕昇编译器的子模块，处理NPU专属优化  |

简单说：**毕昇编译器是“全流程指挥官”，负责跨硬件编译和整体调度；Phoenix是“NPU专项优化兵”，负责算子在NPU上的指令生成和性能调优**，二者协同构成昇腾生态的编译能力核心。

这句话的意思是：该工具链系统提供了一个**软件模拟器（Simulator）**，它可以在PC环境中模拟目标硬件（如NPU、专用芯片）的执行过程，并生成“golden数据”（即标准的、被认定为正确的参考结果）。  

具体来说，在硬件开发早期（如芯片流片前）或软件调试阶段，开发者可能没有实际的硬件设备，此时可以通过这个模拟器在PC上运行算子、模型或程序，模拟硬件的计算逻辑和行为，输出的结果（golden数据）会被作为“基准正确值”，用于后续对比——比如当实际硬件生产出来后，将硬件执行结果与模拟器生成的golden数据比对，验证硬件是否符合设计预期；或用于验证编译器、算子库的输出是否正确（若软件在模拟器上的结果与预期不符，说明软件逻辑存在问题）。  

这句话涉及AI芯片编译器的核心工作流程，而“私有模型IR是否直接包含NPU原语代码”需要从编译器的**多层IR设计逻辑**和**原语的定位**来理解。简单说：**私有IR通常不直接包含NPU原语代码，而是通过多层转换逐步映射到原语**，原语更像是IR优化后的“最终执行单元”。


### 先理清几个关键概念：
- **ONNX**：开放的模型格式，是深度学习框架（如PyTorch、TensorFlow）输出的“通用模型描述”，包含算子（如Conv、MatMul）、张量形状等，但不绑定具体硬件。  
- **私有模型IR**：芯片厂商自定义的中间表示（Intermediate Representation），是编译器将ONNX转换为硬件可执行代码的“中间载体”，通常分为**高层IR**（贴近模型逻辑）和**低层IR**（贴近硬件细节）。  
- **NPU原语（Primitive）**：NPU硬件直接支持的最基本操作（如特定尺寸的矩阵乘、向量加、数据搬移指令等），是硬件能“看懂”并执行的最小单元，类似CPU的汇编指令。  


### 为什么私有IR不直接包含NPU原语代码？
编译器的核心任务是“从高层模型逻辑到硬件执行代码的逐步转换与优化”，这个过程需要多层IR分工协作，而原语是最终映射的目标，而非中间表示本身。具体看：

1. **高层IR：保留模型逻辑，与硬件解耦**  
   刚从ONNX转换来的私有IR（如类似TVM的Relax）属于高层IR，它描述的是模型的计算流程（如“先做卷积，再做激活”），使用的是通用算子（如“Conv2d”），而非硬件原语。  
   这一步的目的是：方便进行与硬件无关的优化（如算子融合、常量折叠），同时兼容不同硬件（如果后续适配新NPU，高层IR无需大改）。

2. **低层IR：逐步贴近硬件，开始关联原语**  
   经过高层优化后，编译器会将高层IR“ lowering ”（降级）到低层IR（如类似TVM的TIR）。低层IR会细化计算细节（如循环拆分、内存布局），此时可能会引入与硬件相关的描述（如NPU的计算单元结构、数据通路），但仍以“抽象操作”为主，而非直接写原语代码。  
   例如，低层IR可能描述“用NPU的MAC阵列做32x32的矩阵乘”，但不会直接写出“MAC阵列的原语指令编码”。

3. **原语：IR优化的最终映射目标**  
   当低层IR完成硬件相关优化（如循环映射到NPU的计算单元、内存布局适配硬件缓存）后，编译器会根据硬件的原语库，将低层IR中的抽象操作**匹配到具体的NPU原语**（如将“32x32矩阵乘”映射到NPU支持的`matmul_32x32`原语）。  
   此时，原语才作为“执行单元”被调用，而私有IR（无论高层还是低层）的作用是“描述如何组合这些原语”，而非包含原语代码本身。


### 举个形象的例子：
假设要编译一个CNN模型：  
1. ONNX输入编译器，先转为**私有高层IR**，描述“输入图像 → Conv2d（5x5 kernel）→ ReLU → MaxPool → ...”（全是通用算子）。  
2. 高层IR优化：将“Conv2d+ReLU”融合成一个算子，减少数据搬运。  
3. 转为**私有低层IR**，描述“融合后的算子需要用NPU的2个MAC阵列并行计算，输入数据从DDR搬到L1缓存，按16x16块拆分计算”（开始关联硬件结构，但仍不直接写原语）。  
4. 最终，低层IR被翻译为**NPU原语序列**（如`load_16x16`（从L1读数据）→ `mac_16x16`（计算）→ `store_16x16`（存结果）），这些原语才是硬件真正执行的代码。  


### 总结：私有IR与NPU原语的关系
| **阶段**       | 核心载体       | 是否包含NPU原语代码？ | 作用                          |
|----------------|----------------|----------------------|-------------------------------|
| ONNX转换后     | 私有高层IR     | 否（用通用算子）      | 保留模型逻辑，做跨硬件优化    |
| 优化后降级     | 私有低层IR     | 否（关联硬件结构）    | 适配硬件特性，准备映射原语    |
| 最终代码生成   | NPU原语序列   | 是（硬件可执行）      | 作为最终执行单元，驱动NPU运行 |


简言之，私有IR是“编译过程中的中间蓝图”，而NPU原语是“蓝图落地后的建筑材料”——蓝图描述如何组合材料，但不会直接包含材料本身。这种分层设计让编译器既能灵活优化模型，又能适配不同硬件细节，是AI芯片工具链的通用逻辑。

你观察到的“芯片厂编译器输出XMM模型文件”，与“编译器后端生成NPU指令”并不矛盾——**XMM是一种封装了NPU指令的“模型容器格式”**，它本质上是将编译器生成的NPU指令、模型元数据（如输入输出形状、数据类型）、权重参数等信息打包后的文件，方便存储、传输和加载执行。


### 具体来说，XMM模型文件的构成通常包括：
1. **核心：NPU指令序列（机器码）**  
   这部分正是编译器后端生成的“NPU具体要执行的指令”，包含计算指令（如MAC阵列的乘加原语对应的机器码）、数据搬移指令（如从DDR到片上缓存的加载/存储指令）、控制指令（如分支、同步指令）等，是模型能在NPU上运行的“执行核心”。

2. **模型元数据**  
   描述模型的基本信息：输入输出张量的形状、数据类型（如FP16、INT8）、内存布局（如NHWC/NCHW）、batch_size支持范围等。这些信息用于NPU驱动在加载模型时分配内存、校验输入合法性。

3. **权重/常量数据**  
   神经网络中的固定参数（如卷积核权重、偏置）会被量化、压缩后存储在XMM文件中，加载时会被搬运到NPU的权重存储器（如片上SRAM或外部DDR），供指令执行时调用。

4. **辅助信息**  
   可能包含版本号（用于兼容性校验）、签名信息（防止篡改）、性能优化参数（如推荐的并行度配置）等，方便工具链或驱动进行加载优化。


### 为什么编译器不直接输出“裸指令”，而要封装成XMM文件？
- **工程实用性**：裸指令（二进制流）难以单独管理，而XMM这样的容器格式可以将指令、参数、元数据“打包”，方便开发者通过文件名、版本号识别模型，也便于存储和传输。  
- **加载效率**：NPU驱动加载模型时，需要先解析元数据（如输入形状）来配置硬件资源（如内存分配），再加载指令和权重。XMM的结构化格式让这个过程更高效，避免手动处理零散的指令和数据文件。  
- **兼容性与安全性**：通过统一的文件格式（如XMM），芯片厂商可以定义版本兼容规则（如高版本驱动支持低版本XMM），同时加入校验机制（如哈希值）防止文件损坏或篡改。  


### 类比理解：
这就像PC上的“可执行文件（.exe）”——.exe文件不只是CPU指令的裸二进制流，还包含程序入口地址、资源文件（图标、字符串）、依赖库信息等，但核心仍是CPU可执行的指令。  
同理，XMM文件是NPU的“可执行模型包”，**外层是方便管理的容器，内层核心是编译器生成的NPU指令**，加载到NPU时，驱动会解析XMM，提取出指令序列并调度硬件执行。


### 总结：
XMM模型文件是芯片厂编译器输出的“最终交付形式”，它**包含了NPU要执行的全部指令**（作为核心内容），同时通过封装元数据、权重等信息，解决了模型管理、加载、兼容等工程问题。你看到的“XMM文件”与“NPU指令”是“容器”与“内容”的关系，前者是后者的实用化封装。

## runtime框架
是的，在多数AI芯片的软件栈中，**runtime框架（运行时框架）的核心输入之一就是XMM这类模型文件**。Runtime的核心职责是“加载模型、管理硬件资源、调度指令执行”，而XMM文件作为编译器输出的“可执行模型包”，正是Runtime与硬件交互的关键载体。


### Runtime框架如何使用XMM模型文件？
其典型工作流程如下：  
1. **模型加载（Load）**：  
   Runtime通过专用API（如`runtime_load_model("model.xmm")`）读取XMM文件，解析其中的元数据（输入输出形状、数据类型）、提取NPU指令序列和权重参数，并将权重搬运到NPU可访问的内存（如片上缓存或DDR）。  

2. **资源初始化**：  
   根据XMM中的元数据，Runtime向NPU驱动申请硬件资源（如计算单元、内存块、DMA通道），并配置硬件状态（如精度模式、并行度）。  

3. **执行调度（Execute）**：  
   当输入数据就绪时，Runtime将输入数据搬运到指定内存地址，然后向NPU发送XMM文件中提取的指令序列，触发硬件执行计算。执行过程中，Runtime会监控硬件状态（如是否完成、是否报错）。  

4. **结果返回**：  
   计算完成后，Runtime从NPU的输出内存中读取结果，返回给上层应用（如深度学习框架或业务系统）。  


### 为什么Runtime需要以XMM为输入？
- **标准化交互**：XMM文件是编译器与Runtime之间的“标准化契约”——编译器按格式输出模型，Runtime按格式解析，双方无需关心对方的具体实现细节（如编译器如何优化指令、Runtime如何管理硬件）。  
- **完整性保障**：XMM包含执行所需的全部信息（指令、权重、元数据），Runtime无需依赖其他文件即可完成加载和执行，简化了部署流程。  
- **硬件适配**：XMM中的指令是针对特定NPU的，Runtime通过解析XMM可以确认模型是否与当前硬件兼容（如指令集版本是否匹配），避免执行错误。  


### 类比：如同PC上的“操作系统+exe文件”
- XMM文件类似Windows的`.exe`或Linux的ELF可执行文件，包含程序（模型）的执行指令和资源。  
- Runtime框架类似操作系统的“进程调度器”，负责加载可执行文件、分配CPU（NPU）资源、运行程序并返回结果。  
runtime框架的**单算子执行**本质是通过“预编译的算子库”实现的——虽然接口输入只有“数据”和“指令名称”（如算子类型，`conv2d`/`matmul`等），但对应的NPU原语代码或指令序列早已被提前编译并存储在**算子库**中，runtime只需通过“指令名称”索引到预编译的执行逻辑，再结合输入数据完成调度即可。


### 具体实现流程拆解：
单算子执行的核心是“**预编译+索引调用**”，避免了每次执行都需要编译原语代码的过程，具体步骤如下：


#### 1. **离线预编译：算子库的构建**  
芯片厂商会提前将常用算子（如卷积、矩阵乘、激活函数等）通过编译器编译为NPU可执行的指令序列（包含原语代码对应的机器码），并按“算子名称”分类存储在**算子库**（类似动态链接库`.so`或专用格式的算子集合）中。  
- 例如，`matmul`（矩阵乘）算子会被预编译为：`load_16x16`（加载数据）→ `mac_16x16`（调用MAC阵列原语）→ `store_16x16`（存储结果）的指令序列，与该算子的功能绑定。  
- 这个过程和“编译器将ONNX转为XMM模型”类似，只是对象从“完整模型”变成了“单个算子”，输出的指令序列被打包到算子库而非XMM文件。  


#### 2. **单算子接口的调用逻辑**  
当用户通过接口（如`runtime_run_op(handle, "matmul", input_data, output_data)`）调用单算子时，runtime的工作流程是：  
- **Step 1：索引算子库**  
  通过接口传入的“指令名称”（如`"matmul"`），在预编译的算子库中找到对应的**预生成指令序列**（包含NPU原语对应的机器码）。  
  这里的“指令名称”本质是算子的“唯一ID”，用于匹配预编译的执行逻辑。  

- **Step 2：数据准备与搬运**  
  根据算子的元信息（预存在算子库中，如输入输出数据类型、内存布局），将用户传入的`input_data`从主机内存搬运到NPU可访问的地址（如片上缓存L1、全局内存DDR）。  

- **Step 3：调度NPU执行指令**  
  runtime将算子库中索引到的指令序列发送给NPU，并配置硬件资源（如激活对应的MAC阵列、设置数据通路），触发NPU执行计算。  

- **Step 4：返回结果**  
  计算完成后，runtime将NPU输出内存中的结果（`output_data`）搬运回主机，返回给用户。  


### 为什么接口不需要输入NPU原语代码？
- **预编译复用**：原语代码对应的指令序列已提前由编译器生成并存储在算子库中，无需用户重复提供。这就像调用C语言的`printf`函数时，无需传入`printf`的底层实现代码——函数库已经包含了其机器码。  
- **简化接口设计**：用户只需关注“调用什么算子”和“输入什么数据”，无需了解底层原语细节（如`mac_16x16`的参数格式），降低了使用门槛。  
- **性能优化**：预编译的指令序列经过编译器深度优化（如原语组合、指令调度），比动态生成原语代码更高效，适合高频调用的单算子场景。  


### 类比理解：像调用“函数库”一样调用算子  
- 算子库 ≈ 编程语言的“标准函数库”（如C的`stdlib`），其中预存了“函数实现”（算子的指令序列）。  
- 单算子接口的“指令名称” ≈ 函数名（如`printf`），用于索引库中的实现。  
- runtime ≈ 函数调用的“执行器”，负责找到函数实现、传参、执行并返回结果。  


### 总结  
runtime的单算子执行依赖**预编译的算子库**：厂商提前将算子的原语代码编译为NPU指令并入库，接口通过“指令名称”索引这些预生成指令，结合输入数据完成调度。这种设计既避免了实时编译的开销，又简化了用户接口，是平衡易用性和性能的典型方案。

### 总结
XMM模型文件是Runtime框架与编译器、硬件之间的“桥梁”，Runtime通过加载和解析XMM文件，将编译器生成的NPU指令“翻译”为硬件的实际执行动作，最终完成模型的推理计算。因此，**XMM是Runtime框架的核心输入之一**，是模型从“编译产物”到“实际运行”的关键载体。

## 模拟器

### 该模拟器与CModel（C语言模型）的区别：  
两者都是硬件开发中用于功能验证的工具，但定位和用途有显著差异，核心区别如下：  


| **维度**       | **Simulator（模拟器）**                          | **CModel（C语言模型）**                          |
|----------------|------------------------------------------------|------------------------------------------------|
| **本质**       | 基于硬件规范实现的“执行环境模拟器”，更贴近实际运行场景 | 用C/C++语言编写的“硬件功能抽象模型”，是硬件逻辑的“说明书” |
| **核心用途**   | 供软件开发者使用：模拟硬件执行过程，生成golden数据，验证算子、模型、编译器输出的正确性 | 供硬件设计者使用：定义硬件的功能规范（如计算逻辑、接口行为），验证硬件逻辑设计（如RTL代码）是否符合预期 |
| **抽象层次**   | 更接近“用户视角”，模拟硬件的外部行为和执行流程，可能包含简化的时序或资源模型 | 更接近“硬件视角”，抽象描述硬件内部的功能逻辑（如MAC阵列的计算规则、数据通路），通常不涉及具体时序 |
| **使用阶段**   | 偏后期：硬件规范基本确定后，用于支持软件栈（编译器、算子库、应用）的开发和调试 | 偏早期：硬件设计初期，用于定义功能边界，指导硬件RTL设计（相当于硬件的“功能契约”） |
| **典型输出**   | 程序/算子的执行结果（golden数据）、性能预估（如耗时） | 硬件功能的逻辑验证报告（如“输入A时，输出是否符合预期B”） |  


### 举例说明：  
假设开发一款NPU：  
- 硬件团队先编写**CModel**，用C语言定义“NPU的MAC阵列如何做矩阵乘”“数据如何从内存传入计算单元”等核心功能逻辑，作为硬件设计的“功能标准”。  
- 基于CModel的功能规范，工具链团队开发**Simulator**，模拟NPU的完整执行流程（包括数据加载、计算、结果输出）。软件团队用这个Simulator运行卷积算子，得到的输出就是“golden数据”。  
- 当NPU的RTL代码设计完成后，用CModel验证RTL逻辑是否符合功能规范；当芯片流片后，用Simulator生成的golden数据验证实际NPU的执行结果是否正确。  


简言之：**CModel是硬件功能的“定义者”，Simulator是硬件行为的“模拟器”**，前者服务于硬件设计验证，后者服务于软件开发调试，二者协同保障硬件和软件的一致性。
## 量化
### 对称量化
有符号数

### 非对称量化
无符号数

### moe算子阶段做混合精度量化
PTQ量化和QAT导入



## 业界存储形态
bm1n1m0n0性能最好，功耗最低
你提到的这些数据排布（如NCHW、NHWC、NC1HWC0、BMN、bmn1n1m0n0）本质上是**多维张量在内存中的存储顺序规则**，用于规范不同维度的数据如何映射到连续的内存地址中。它们的设计通常与硬件架构（如GPU、专用AI芯片）的计算效率密切相关，不仅限于CV领域，在LLM等其他AI场景中也可能因计算需求或硬件约束而存在类似的多维维度划分。


### 【AI系统】昇腾数据布局转换


**简介：** 华为昇腾NPU采用独特的NC1HWC0五维数据格式，旨在优化AI处理器的矩阵乘法运算和访存效率。此格式通过将C维度分割为C1份C0，适应达芬奇架构的高效计算需求，支持FP16和INT8数据类型。此外，昇腾还引入了NZ分形格式，进一步提升数据搬运和矩阵计算效率。AI编译器通过智能布局转换，确保在不同硬件上达到最优性能。

NHWC 的数据排布方式更适合多核 CPU 运算， NCHW 的数据排布方式更适合 GPU 并行运算。那么接下来让我们了解一下在华为昇腾的 NPU 中，这种特征图的存储方式。

> 截止到 2024 年，华为昇腾在私有格式的数据处理和特殊的数据形态越来越少，主要是得益于 AI 编译器和软件的迭代升级，更加合理地兼容业界主流的算子和数据排布格式。



#### 昇腾数据排布格式


数据排布格式的转换主要是将内部数据布局转换为硬件设备友好的形式，实际在华为昇腾的 AI 处理器中，为了提高通用矩阵乘法运算和访存的效率，一般既不选择 NHWC，也不选择 NCHW 来对多维数据进行存储。

这里我们将华为昇腾的数据排布作为一个案例，这种多维数据统一采用 NC1HWC0 的五维数据格式进行存储，具体的含义是将数据从 C 维度分割成 C1 份 C0。

如下图中所示，下图中将 N 这个维度进行了省略，原先的红色长方体展现的是 CHW 三个维度，将它在 C 维度分割成 C1 个长方体，每个长方体的三个维度为 C0HW，而后将这 C1 份长方体在内存中连续排列，此处的 C1=C/C0，如果不能除尽则向上取整，对应着上半篇中的内存对齐，也就是通道现在变成了 C0 个，其中 C0 对于 FP16 类型为 16，对于 INT8 类型则为 32，这部分数据需要连续存储。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_26bcf00faa614dffa87e4d89a7bf8177.png?x-oss-process=image/resize,w_1400/format,webp)

这样子的数据排布我们从硬件的角度来进行分析，华为的达芬奇架构在 AI Core 中特意优化了矩阵乘法单元，矩阵计算单元可以快速完成两个 16x16 矩阵的相乘运算，等同于可以在极短时间内进行 163\=409616^3=4096\\quad\\quad\\quad 个乘加运算，并且可以实现 FP16 的运算精度，也就是说其可以针对 16 个 FP16 类型的数据进行快速的计算。这也就是我们对 C0 在 FP16 类型取 16，INT8 类型取 32 的部分原因。

下面我们来介绍一下如何转换出 NC1HWC0 数据格式，即将 NHWC 转换为 NC1HWC0 数据格式。具体操作：

1.  将 NHWC 数据在 C 维度进行分割，变成 C1 份 NHWC0。
    
2.  将 C1 份 NHWC0 在内存中连续排列，由此变成 NC1HWC0。
    

pytorch 中代码如下所示

    Tensor.reshape([N, H, W, C1, C0]).transpose([0, 3, 1, 2, 4])
    

将 NCHW 转换为 NC1HWC0 数据格式

    Tensor.reshape([N, C1, C0, H, W]).transpose([0, 1, 3, 4, 2])
    

#### Fractal Z & NZ 格式


ND 格式 （N-Dimension），是神经网络中最常见且最基本的张量存储格式，代表 N 维度的张量数据。

为了在达芬奇架构中更高效的搬运和进行矩阵计算，引入一种特殊的数据分形格式，NZ 格式。  
如下图所示，我们以 4\*4 的矩阵来进行举例，按照 NZ 格式数据在内存中的排布格式为\[0，1，4，5，8，9，12，13，2，3，6，7，10，11，14，15\]，按照 ND 格式数据在内存中的排布格式为\[0， 1，2，3，4，5，6，7，8，9，10，11，12，13，14，15\]。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_30e2554ff11746b6b4af0ef2447be498.png?x-oss-process=image/resize,w_1400/format,webp)

如下图所示，NZ 分形操作中，整个矩阵被分为（H1 \* W1）个分形，分形之间按照列主序排布，即类比行主序的存储方式，列主序是先存储一列再存储相邻的下一列，这样整体存储形状形如 N 字形；每个分形内部有（H0 \* W0）个元素，按照行主序排布，形状如 Z 字形。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_93eccef2dab545ada3eefab137462d4a.png?x-oss-process=image/resize,w_1400/format,webp)

下面我们详细地对 NZ 格式（也被称为 NW1H1H0W0 格式）具体在内存中存储的维度优先顺序进行展开。先对一个分形内部进行行主序存储，再在一个完整矩阵中以分形宽度为划分，进行列主序存储，再依次对相邻的下一个矩阵进行存储。即此方式先按 W0 方向存储，再按 H0 方向存储，接着按照 H1 方向存储，随后按照 W1 方向存储，最后按 N 方向存储，直到所有数据存储完成。

下面我们介绍一下如何将 ND 数据格式转换为 NZ 数据格式：

将 ND 转换为 NZ 数据格式

    (..., N，H, W )->
    pad->
    (..., N, H1*H0, W1*W0)->
    reshape->
    (..., N, H1, H0, W1, W0)->
    transpose->
    (..., N, W1, H1, H0, W0)
    

其中 pad 为平铺操作，reshape 将张量进行拆分，形状重塑，transpose 为转置操作。

除了 ND 和 NZ 格式，还有其他数据格式，如下图所示，图中最左侧小 z 大 Z，即为 ND 格式示意图，块内按照行排序，块间也按照行排序，常用于特征图的数据存储。

图中中间部分为小 n 大 Z，块内按照列排序，块间按照行排序，常用于权重的数据存储。图中右侧部分为小 z 大 N，即为 NZ 格式示意图，块内按照行排序，块间按照列排序，常用于卷积结果的输出。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_64a1295788ff45bb99a702105ede33a3.png?x-oss-process=image/resize,w_1400/format,webp)

#### AI 编译器布局转换算法


了解了基础知识与部分硬件中应用后，我们来了解一下在 AI 编译器中如何对数据布局进行转换优化。

首先，我们转换数据布局的目的是将将内部数据布局转化为后端设备（硬件）友好的形式，我们需要做的是尝试找到在计算图中存储张量的最佳数据布局，然后将布局转换节点插入到图中。

但其中有个需要十分注意的地方，布局转换也是需要很大的开销的，一旦涉及布局转换，就会有 I/O 操作，其产生的代价能否比的上数据格式转换后带来的性能优化也是需要我们重点考虑的部分。

具体地来说，比如 NCHW 格式操作在 GPU 上通常运行得更快，所以在 GPU 上转换为 NCHW 格式是较为有效的操作。

一些 AI 编译器依赖于特定于硬件的库来实现更高的性能，而这些库可能需要特定的布局，比如华为昇腾的 AI 编译器就依赖于 CANN 库，其中的特定布局我们在上文中已经提到。

同时也有许多设备需要配备异构计算单元，比如手机，其 SOC 中有丰富的 IP，arm 端侧的 GPU 还有 ISP 以及 DPU 等一系列不同计算单元。不同的单元可能需要不同的数据布局以更好地利用数据，这就需要 AI 编译器提供一种跨各种硬件执行布局转换的方法。

下面我们来看看数据转换具体是如何操作的。如下图所示，这两个都是数据转换的算子，数据转换我们在这里用 CASTDATA 算子来表示，左侧输入的数据格式为 NHWC，输出的数据格式为 NCHW，那么就需要一个数据转换算子节点来将数据格式由 NHWC 转换为 NCHW，右侧则相反过来，此处不再赘述。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_d3b21fdad2474be680a44e8fa0819052.png?x-oss-process=image/resize,w_1400/format,webp)

接下来，我们来看略复杂一些的数据转换。如下图所示，首先像最左侧，此处我们两个算子使用的数据格式与输入输出时都相同，为 NCHW，那么此时 AI 编译器中就不需要加入数据转换节点。

中间部分，输入的数据格式为 NCHW，算子一需求的数据格式为 NHWC，需要在两者之间插入一个 CASTDATA NCHW TO NHWC 算子进行数据转换，算子二格式也为 NHWC，数据格式相同，不需要转换，输出的数据格式为 NCHW，那么在算子二到输出间需要插入一个 CASTDATA NHWC TO NCHW 算子进行数据转换。

最右侧的图中，输入的数据格式为 NCHW，算子一需求的数据格式为 NHWC，需要在两者之间插入一个 CASTDATA NCHW TO NHWC 算子进行数据转换，算子二格式为 NCHW，需要在算子一到算子二之间插入一个 CASTDATA NHWC TO NCHW 算子进行数据转换。输出与算子二数据格式相同，不做额外处理。

![](https://ucc.alicdn.com/pic/developer-ecology/yysinyik4knec_262848265bf04a148a478293ea482a10.png?x-oss-process=image/resize,w_1400/format,webp)

首先我们来讲解一下训练场景下 AI 编译器的例子，例如 1×1 的卷积常常使用 NHWC 的数据格式，而如果到后面使用的是 3×3 的卷积常常使用 NCHW 的数据格式，AI 编译器此时能够感知上下文，得到这些存储格式的信息，这时候 AI 编译器就会根据具体的数据格式插入需要的转换算子，并且整个过程不会改变原先原计算图的其他内容。

再假设此时我们使用的是三个连续的 1×1 的卷积算子，使用的都是 NHWC 的数据格式，算子之间的数据格式是一样的，那么此时 AI 编译器需要取消掉多余的转换算子。

接着我们来讲解一下推理场景下 AI 编译器与训练场景下有什么不同，其中较大的区别是与会有在权重数据进行布局的转换。假设训练时使用的是 GPU 对神经网络进行训练，但是推理的时候会在更多的场景下进行使用，比如手机上进行推理，手机上较多使用的是 CPU，其进行推理时与在 GPU 上进行训练时的权重数据的布局可能会有所不同，那么此时就需要 AI 推理编译器插入一个权重布局的转换。

### 达芬奇架构四大计算单元详解
https://developer.aliyun.com/article/1642737?spm=a2c6h.13262185.profile.13.5a0376b4ihH2o3


## npu精度
### 《NPU与GPU间为什么会有精度误差？！》笔记
#### 1. 数学层面：浮点数运算的特性
- 浮点数不满足数学上的交换律和结合律，不同的计算顺序和分组方式会导致结果差异。
  - 举例：0.1、0.2、0.3、0.4、0.5、0.6通过不同分组累加，结果分别为2.1和2.0999996。
- 结论：即使相同输入和芯片，因数据切分策略、交换/结合方式不同，结果也可能不同；不同芯片间更难等价。


#### 2. 硬件层面：实现方式的差异
- **浮点计算单元设计不同**：不同厂商的FPU（浮点计算单元）在流水线设计、乘法器/加法器位宽优化、数据处理方式上有差异。
  - 英伟达GPU（如A100、H100）采用复杂累加器，Tensor Core针对矩阵优化。
    - ![alt text](images/image.png)
    - ![alt text](images/image-1.png)
  - 华为昇腾NPU（如A1、A2）的Cube Core专为AI矩阵计算优化，属DSA架构（领域特定架构）。
    - ![alt text](images/image-2.png)
    - ![alt text](images/image-3.png)
- **精度格式缺乏统一标准**：IEEE 754标准未涵盖FP16、BF16、FP8等AI常用精度格式，厂商可自主实现（如英伟达FP8与华为Half8），导致硬件天然存在差异。


#### 3. 软件与算法层面：实现策略的差异
- **底层数学库不同**：英伟达依赖cuBLAS，昇腾依赖CANN算子库，矩阵分块（tiling）、循环展开等策略不同，影响浮点数累加顺序。
- **编译器优化差异**：英伟达nvcc与昇腾编译器对指令重排、融合的处理不同，进一步影响计算结果。


#### 4. 并行计算层面：非确定性
- 线程调度顺序、数据同步时机存在不确定性，不同硬件（如GPU的多线程调度与NPU的单指令多数据架构）的数据更新时序不同，引入误差。


#### 5. 累积误差：舍入误差的叠加
- 浮点数运算中的舍入误差会随计算步骤累积，且因缺乏统一标准，误差范围难以完全一致。


#### 总结
- **误差的普遍性**：GPU不同代产品（如A100与H100）、GPU与NPU、不同NPU产品间均存在精度误差，NPU因架构跨度大，误差可能更明显。
- **误差的可控性**：精度误差通常在小数位后千位级别，可控制在合理范围，不影响大模型性能（如DeepSeek用FP8训练效果良好）。
- **模型的适应性**：AI模型泛化性强，低精度计算不会显著降低性能，无需追求绝对精度一致。






## 代办
### 动态与静态图
动态moe


