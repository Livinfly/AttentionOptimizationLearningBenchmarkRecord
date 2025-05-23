# 调研LLM和DiT相关的Kernel优化工作/library：

## FlashInfer

https://github.com/flashinfer-ai/flashinfer；https://arxiv.org/pdf/2501.01005

### 动机

-   LLM应用表现出多样化的工作负载模式和输入动态
-   现代硬件实现需要对attention算子进行定制化



### 贡献

-   引入了一种块稀疏格式，以解决KV-Cache存储异构性的问题，并且当前主流的KV-Cache的存储方式，都可以在块稀疏下得到统一
-   提供了一个可定制的attention模板，支持不同的attention变体
-   建立了一个动态负载均衡调度框架，以应对输入动态性



### KV-Cache

引入了一种块稀疏格式Block Compressed Sparse Row (BSR)，以解决KV-Cache存储异构性的问题，并且当前主流的KV-Cache的存储方式，都可以在块稀疏下得到统一。

可组合格式composable formats多种稀疏矩阵格式，减少碎片，提高利用率

允许在共享前缀和后续后缀之间解耦注意力计算，加快并行解码

（单个大 Br 值，提高共享内存shared memory，寄存器register的利用率）



### 计算抽象 Compute Abstraction

Global -> shared memory 的稀疏块的装载

实现多种分块大小的FA2

attention 变体的模板JIT高效实现（比如RoPE和attention的kernel fusion等，可以大大降低实现高效kernel的难度）



### 负载均衡动态调度

根据查询/输出、键值长度，计算出最大chunk，CPU产生调度信息，分别得到「归约图Reduction Map」和CTA work queue，最后计算，归约。



### 评估结果

-   在不是固定长度的 Uniform、Skewed 分布下，由于调度负载均衡调度 Kernel-level 的表现比 constant 的优势更加明显。

-   自定义注意力kernel，flashinfer融合的RoPE 与 未融合的比较，进步明显（不过没有和manual 融合的比较），不过足够说明这种的注意力kernel模板的必要性，为实现高性能算子实现简单的接口。
-   并行生成，composable formats



### 总结

1. 解决前面的多种kv-cache存储的异构问题，提出块稀疏形式，还能进一步提高内存利用率，composable formats，做些之前的通用技术的benchmark证明性能
2. 高性能的定制化attention模板，支持任意block大小，多种tiling大小，能适应不同架构
3. 动态负载均衡调度，CPU调度、GPU计算解耦，对动态性输入的处理性能提升，对 fa decoding batching 的进一步缓解（分块 query tile 的 KV 调度，这样之间是可以并行的，所以对单query，加速也是适用的）

细节都需要再去看源码了。



### 后续可能研究方向

flashinfer 本身拓展到训练框架，需要 attention backward kernel 模板

基于 flashinfer 的调度与计算解耦，当前调度算法比较朴素，各个场景的调度算法，不用考虑调度问题的计算方法

多层块稀疏组合参数选择



### 优化思路

对同个 query tile 需要的 kv 负载动态调度



### 源码阅读

1.   JIT 编译模板和代码生成逻辑

     -   在.cuh文件中写好CUDA模板，使用模板参数，用 Jinja2 做模板生成代码，再用 PyTorch 的 `torch_cpp_ext` 动态编译，再TORCH_LIBRARY_FRAGMENT注册内核，torch.ops调用注册的内核
     -   通过@register_custom_op装饰器等方式，把Python函数注册到底层 C++/CUDA （再具体就略过了）
     -   流水线、PTX 指令等方式加速
2.   BSR 和 composable formats

     -   纠正理解上的错误，BSR 是用来稀疏表示的，KV Cache 存储本身还是 Paged/Ragged 存储，BSR 加速稀疏注意力的计算，提高内存访问效率。
     -   源码中好像没有 composable formats 的实现，论文最后有提到实际效果中有时可能有副作用，猜测是实际共享前缀比较短，因此没有放上去（？）
3.   动态负载均衡调度

     -   在host端规划好调度元数据，device端计算




## Sage & Sparge系列

https://github.com/thu-ml/SageAttention；https://github.com/thu-ml/SpargeAttn；https://arxiv.org/pdf/2411.10958；https://arxiv.org/pdf/2502.18137

### SageAtten2

#### 动机

-   INT4 比 INT8 matmul 快，利用 INT4 tensor core
-   FP16 matmul 加速只在少部分型号上支持，推广加速方法适用硬件范围



#### 贡献

-   通过硬件指令（PTX mma）反推量化分组策略，提出 per-thread 量化方案，对 latency 和 accuracy 做了最优 trade-off
-   通过平滑Q K等方案，提升注意力计算精度
-   提出二级累加策略，改善硬件指令设计的精度问题，同时也提出平滑V来提升原硬件指令精度的方法



#### Smooth Q, K

在 SageAtten 的 smooth K 的基础上，继续提出 smooth Q（根据所有 token 的 Q K 高度相似）

通过 softmax 不变量特性，省略不必要的计算

之前经典的 activation-weight 方法的 per-channel 并不适用于 query-key MM，因为 query-key MM 是逐 token 对计算的，per-channel 对 kernel 不友好，即不能一次性dequant，增大开销

对于 attention 计算来说，之前的 smoothquant，会面临 Q，K 都有outlier的时候，此时 weight-activation trade 效果不好

（所以smoothquant是怎么处理attention这部分的 trade 的？）

根据源码，**smoothquant** 会取 QKV 对应 channel 的最大值作为衡量标准，统一得出一个缩放参数 s。

（不确定进行实现的时候，是对称量化还是非对称量化，对称量化和非对称量化的优缺点分别是？减去 mean 是因为采用了对称量化吗？）

根据源码，**对称量化**，对称量化计算简单，速度更快，精度可能相对低一些，但对对称分布又还好，减去 mean 后，同时也让对称量化的精度更高。

（weight，activation attention的数值分布是怎么样的）



（per-token是对整个token维度量化一次，还是每个head的量化一次？）

根据源码，是按照不同head，分别去 per-token。



#### INT4 Per-thread quantization

根据 PTX mma 指令，让同个thread用相同的量化参数

借助**附录A.5.**和**源码**。

目标或原因，利用 tensor core INT4 mma，高效反量化。

1.   为什么 INT4 mma 是 m16n8k64

     -   硬件资源，乘法器、加法器等资源，形状关系计算量；面积、功耗、散热。
     -   一个MMA是由一个Warp协同执行，M和N需要高效映射到32个线程上。
     -   K和计算强度、内存带宽、寄存器容量等相关。
     -   INT4 **mma.m16n8k64**，同时也有 INT8 **mma.16n8k32**，由于前面说的原因k相比会小一些，由于INT4能支持的k更大所以 INT4 会更快

2.   由 INT4 mma 反推 Q 和 K 的量化组，高效反量化

     -   每个线程，执行$D_{frag}=(δ_Q×A_{frag})×(δ_K×B_{frag})+C_{frag}$ 时，只要加载两次 scale 就好了。

     -   一个 Warp 32线程，结果 m16n8 有 16\*8 = 128 个元素，所以，每个线程需要存储 4 个 INT32。
     
     -   具体地，Memory layout 论文 Figure 17 给出，一个线程存储 (0, 0), (0, 1), (8, 0), (8, 1) 的 INT32。
     -   从 Memory layout 也可知道，为什么 Q 分为 8 个量化组，K 分为 4 个量化组同时相邻的两个放入同个量化组中。
     -   以最后存在一个线程中的 INT32 的个数，来确认行列（QK）共享量化参数个数，在根据自己设定的 WARPQ/K 论文中示例 32/64，共享。



#### V FP8 quantization

适用于更多GPU架构加速



#### FP22 Accumulator

二级累加策略，另外可选方案。

（看着他里面实验结果 4bit 和 8bit 还是差些，8bit和全精度就不怎么差了，如果4bit再提高些精度，可能效果可以再好一点？）



#### 总结

聚焦于Attention的QKV参数量化加速与精度，以**利用 INT4 tensor core 加速**与**增加不同架构加速方法的适用性**为目标，基于Q, K只有少数不同的特点，进一步提出 **Smooth Q, K** 来保持精度，由硬件指令反推量化分组策略 **Per-thread** 平衡 accuracy 和 latency，指出 mma(fp32fp8fp8fp32) 可能导致的精度问题，并提出二级累加策略和另一个可选方案。

在细节实现上，用 softmax 的特性规避了不必要的计算。

稠密attention。



#### 后续可能研究方向

INT4虽然效果不错，但是和INT8与全精度还是有区别，INT8和全精度已经是差不多了。

能不能搭配 INT4 + 可能会加些overhead 的 对每个head的token维度量化？



#### 优化思路

都是基于硬件特性来的，减少不必要的 overhead，同时根据 attention 的 QKV 的特点，定制了量化维度，与量化精度提升方法，来降低延迟、提高精度。

和前面工作不一样的点在于对 attention 中计算的特点去特别设计，无论是 per-thread，还是 Smooth Q, K。



#### 源码阅读

1.   Per-thread Quantization

     主要还是对 Per-thread 划分的理解，见前面 **per-thread** 章节的内容，源码上看，主要是 thread_per_block 是最后需要存储的量化参数的个数，即符合一个线程使用一组量化参数的设计。

     具体地，BLKQ/K 应该是为了对应后续需要做的 MMA 的块划分（一个warp处理多少）对齐（不确定），WARPQ/K （必须32倍数）是在多少个WARP Q/K 共享这个量化参数。

2.   Smoothing Q and K

     -   `csrc/fused/fused.cu <func>quant_per_block_int8_fuse_sub_mean_cuda` 

         kernel 实现统一在`<func> QuantInt8Kernel` 中，元编程当作模板类

     -   INT4 量化在 `triton/quant_per_thread.py` 实现，不过好像没有放出对应 INT4 smooth 的代码，不过确实不是论文的优化重点就是了（？

3.   Two-level Accumulation Strategy PV

     -   在`qattn/attn_utils.cuh <func>compute_fp8_sv_inst_buf ` 中，fk = 0 和 fk > 0 分开写循环，分别展开

         **fk = 0** 时，需要 mma KInit，**fk > 0** 时，是 mma kInplaceUpdate，减少不必要的分支判断，或者不同操作对寄存器的使用不同等原因。

5.   架构实现差异

     -   SM8.0，基础的 INT8 和 FP16 MMA，`cp.async`。
     -   SM8.9，**FP8 MMA** PV，两级累加。
     -   SM9.0，**TMA** 优化内存拷贝，**WGMMA** 提升 MMA 计算效率。



### SpargeAtten

#### 动机

-   现有的方案多是为特定模型、任务，提出注意力的稀疏模式，在其他任务上效果不好，不够通用
-   现有方案效果和性能不好，或者需要较高代价



#### 贡献

-   通用的稀疏掩码预测算法
-   稀疏在线 softmax 算法



#### Sparse FlashAttention

基于 FA 的分块策略，每个 qk对 块有 块掩码 block masks，为 1 的才要计算。



#### token compression & sparse prediction

虽然 attention map 因模型不同各有不同，但共同特征是，相近token高度相似。



把 Q K 中相似的 token 块各自合并，得到代表 token，

具体是用 **token维度求均值**，也就是可能的作为**代表token**的值 q, k，

同时求**块内余弦相似度**，我们**不信任块内相似度低**的，最后都选择保留原 token（mask 为 1），

不过对于 **K 中相似度低的块**，设为负无穷

（具体是用了哪种 softmax？~~FA还是传统 softmax，传统的话应该是减少误差，影响降低~~），

**FA，在线 sm**

减少因为它可能产生的 幂次最大值m 变化而产生的计算（吗？）

**没有具体说，我觉得在online softmax中，在线的normal 乘 1 和 乘一个倍数，应该还是有些计算差异的（？**

**也有把这一块排除在「自相似度高」的块的外面，保证TopCdf选出的都是自注意力得分高的**

然后 Q 的话，由于没有能减少的计算，就不做处理

计算出压缩后的 attention map $\hat{P}$，保留选择 $\hat{P}$ 大的 (i, j)，因为他们注意力分数高，重要



注意的是，只能压缩高度相似的，否则容易丢失重要信息。

（余弦相似度的道理在哪里？有多少？现在都是按照魔法相似度理解的，bushi）

**查询、值是向量，VSM、embedding等中使用并且效果好的，是个公式-经验-公式的过程。**



#### Sparse Warp Online Softmax

当 P 很小时，PV 可以忽略。

具体地，可以利用 onlnine softmax 中存储的**幂次最大值 m**，

当 rowmax(S_ij) << m_ij 时，P 很小，可以忽略，减少影响少的计算。

和量化操作的搭配，两者是正交的，可以一起使用。



#### HilbertCurve Permutation

为了尽可能增加稀疏度，我们希望出现块内一致性高的情况，

发现调整输入 token 的排列顺序是没关系的，所以重新排列。

空间填充曲线，高维映射到一维序列，最大程度保留局部性，邻近保持。

结合数据先验和计算优化

（感觉先验算法翻找出来的）



#### 实验结果

从实验结果上来看，有些比全注意力还要好，比如LLM，让其关注更有关的信息。

同样，在LLaMa3.1上，序列越长，稀疏度越高，128k有54%

（后面在附录有瞄到关于时间major的，其实专用稀疏注意力还是有发展范围的，不同格式可利用的特性还是有的）

自相似性有效降低了精度误差。



#### 总结

基于前面多个token打包压缩出一个token的工作，发现如果不考虑块内一致性的话，容易丢失重要信息，

提出使用**余弦相似度**去算块内一致性，块内一致性高，同时注意力分数不高，再去压缩，配合 warp 的特点，

合理设计块大小，并且对压缩的token判断skip，同时有把**块内一致性**不高的k对应列赋值负无穷等一系列符合算子特性的trick。

在稀疏注意力下，sparse online softmax 中，略过很小的 P，借用 online softmax 算出的 m，减少计算。

最后在不改变注意力结果的情况下，使用希耳伯特曲线HilbertCurvy先验，改变块的划分，增前局部性，提高稀疏度。

稀疏注意力。



#### 后续可能研究方向

稀疏注意力能提升如LLM的长上下文的效果，让注意力到更需要的地方，可能在训练阶段的attention设计进行改善？

感觉注意力通用稀疏探底了（x

不过**专用稀疏注意力**里，比如视频前后帧，注意力稀疏情况不会差太多，然后前后帧变化幅度，预测个变化方式，感觉是不是有说法？



#### 优化思路

考虑块内一致性与注意力得分，压缩 token，根据input动态变化，通用性高；

根据算子特性，省略不必要的计算与误差，在精度允许范围内，再次增大稀疏度；

用数据先验，改变数据分块顺序，增强局部性。



#### 源码阅读

1.   Sparse FlashAttention

     有用小的模板，去批量实现各个参数设定的cuda文件。

     `SpargeAttn/csrc/qattn/qk_int_sv_f8_cuda_sm90.cuh <func>qk_int8_sv_f8_attn_kernel` 

     1.   Masking of the First Stage

          在CUDA核函数前决策。

          Mg，由于前面处理出来 lut 和 valid_block_num，所以，自然跳过Mg = 0的 K 块

     2.   Sparse Warp Online Softmax

          如果整块中的 local_max_diff 大，那么说明里面有不能忽略的小块，需要计算。

          否则不处理。

2.   Selective Token Compression for Sparse Prediction

     主要逻辑在`SpargeAttn/spas_sage_attn/utils.py <func>get_block_map_meansim` 

     或者 fuse 量化的方法。

     CosSim 的方式并不是论文中的公式，而是先L2归一化，再算余弦相似度，最后求平均。

     数值稳定性更高，目标是一样的，归一化，块相似度。

     包括 Mg 的设计中，还有 attention_sink 指定为重计算的可选项，论文中没有提到。

     （不过一般来说attn sink也会比较高，只是更保险）

     `<func> block_map_to_mask` 把块粒度的稀疏，转化为token粒度，主要是兼容，可视化

     `<func> block_map_lut` 变成Lookup Table，高效访问，同时是累加偏移量。

     （Triton 的写法是块级的抽象，需要用tl.arange + mask去让triton分配线程。）

6.   Hilbert Curve Permutation

      好像并找不到实现，不过加在数据预处理就行（？

7.   CUDA 相关小实现

     对 PTX 指令做 wrapper，同时用元编程的方式，做比如 init, inplace_update 等分类的优化生成。



## STA

https://github.com/hao-ai-lab/FastVideo；https://arxiv.org/pdf/2502.04507

### 动机

-   DiTs 视频生成效果很好，计算昂贵
-   其他方法，理论值很好，但实际结果一般，attn 计算开销太大
-   视频冗余大



### 贡献

-   对 3D locality, head specialization 进行指出、量化
-   提出和 SWA 相比 gpu-friendly 的 STA，提高 attn 计算效率



### Sliding Tiling Attention

由于 Tiled NATTEN 等之前的工作出现的稀疏度没有实际很好的转化为计算量减少，

认为是不做 gpu-friendly 的处理导致的，如 mixed block 实际没有减少计算量，却还要计算额外的 attn mask。

STA 做分块划分的时候，只做整块与整块之间的 attn 计算，去除掉 mixed block，同时保持局部性。



再做了 kernel-level 优化，把线程块分成 **计算 warpgroup** 和 **数据 warpgroup**，

利用 warpgroup 的异步，把**计算块间掩码**和**数据加载**的开销。



应用的时候，先STA Mask Search，设定阈值，考虑**head specialization**，为每个头找合适的 mask

同时，也可以在确定mask之后，选择fine-tune，设定loss attn + final + data



### 实验效果

主要分为 STA 和 kernel-level 异步的优化效果。

STA 的比较，文中使用在 FlexAttention 的基础上实现，与之前的稀疏注意力相比大大提升了 **MFU**，从不到 10% 提升至 41%。

和不用稀疏达成 7.30x，和 Tiled NATTEN 相比，也是 5x

kernel-level 异步的优化效果，在 ThunderKittens 上实现，提升至 10.45x，说明两者相比原先做法能达到有效提升。



### 总结

优化稀疏分块方式，解决了之前工作提高稀疏度，但速度反而变慢的情况，削除了mixed-block掩码计算和低效计算

kernel-level，利用warpgroup异步，overlap 开销



判断时空交替的注意力，问题在于破坏了 3D locality 模式



### 后续可能研究方向

和 caching, distill 等方法结合

对较远的块不选择直接跳过，而保留部分信息，如SpargeAttn合并



### 优化思路

优化稀疏分块方式，解决了之前工作提高稀疏度，但速度反而变慢的情况

kernel-level，利用warpgroup异步，overlap 开销



### 源码阅读

1.   SLIDING TILE ATTENTION

     大多数是边界的设计，思路觉得论文中的讲述已经蛮清楚。

2.   Kernel-level optimizations

     异步加载，生产者-消费者，流水线加载 K::stages，Ping-Pong Buffer (K::stages = 2)、多级流水线

3.   Head specialization and automatic window size configuration

     没有显式提供STA Mask Search，不过伪代码清晰，就是在误差满足的条件下，稀疏度越高的。

     由于Head specialization，就每个head都找一个。

4.   Training and Finetuning

     重点看了下distill的，条件输出和无条件输出结合提高多样性。

     distill 也不完全是论文中的 Loss，只能算是有 data、final，再有adv对抗学习的方法。

     （总体看得比较略了）



## 有些还不太了解的

FA2, 3

Block-Parallel Transformer (BPT)

RadixAttention

CUDAGraph，Persistent Kernel等

Inspector-Executor (IE) 模型

relatedwork，FlashDecoding Split-K，LeanAttention StreamK……

online softmax 源码实现

~~不列了，到时候再看一边，找related work就都是了~~

过了一下的

(online softmax) Online normalizer calculation for softmax，降低memory access 1.33

