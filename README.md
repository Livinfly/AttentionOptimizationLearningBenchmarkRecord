# Benchmark_EXP

默认以 CogVideoX-2b 模型任务为例，测试不同注意力模块的加速情况。
由于使用 RTX 4090 测试，不是 Hopper 架构，不能选择 fa3, sta 测试。
benchmark 代码参考了 sage 的 example


测试对象为：sage2, sparge, sparge_fp8

baseline: torch backend flash_sdp

## 环境

-   Python 3.12(ubuntu22.04)

-   CUDA 12.4

-   PyTorch 2.5.1

    

-   **CPU**: 16 vCPU Intel(R) Xeon(R) Gold 6430

-   **RAM**: 120GB

-   **GPU**: RTX 4090(24GB)

## 分析

由于自己手头 GTX 1660S 有若干特性不能支持，线上租用算力平台不支持 **ncu** 分析，所以主要分析依据是 **nsys**。

下表是从 **CUDA Summary (API/Kernel/MemOps)** 摘出的比较突出的点。

（Warmup 2 轮，Run 1轮，10 \* 2 + 50，标准默认记录总耗时，cudaDeviceSynchronize, cuLaunchKernel 不另外分析了）

| Metrics                   | FA2 (baseline) |  Sage2  | Sparge (fp16) | Sparge (fp8) |
| ------------------------- | :------------: | :-----: | :-----------: | :----------: |
| cudaStreamSynchronize (%) |     41.8%      |  32.9%  |     24.3%     |    23.1%     |
| cudaStreamSynchronize (s) |     112.3s     |  78.5s  |     49.7s     |    46.3s     |
| Attn (%)                  |     24.4%      |  12.0%  |     14.1%     |    12.2%     |
| Attn (s)                  |     65.6s      |  28.6s  |     28.7s     |    24.5s     |
| cudaLaunchKernel (times)  |    139,547     | 190,526 |    188,380    |   194,291    |
| cudaLaunchKernel (s)      |     13.2s      |  14.7s  |     19.4s     |    31.3s     |
| Total (s)                 |     88.4s      |  65.6s  |     67.2s     |    64.1s     |

总的来说，**cudaStreamSynchronize** 的时间都是最长的，说明计算还是慢了，CPU 大部分时间在等待，三种优化方案都有不同程度的提升。

占计算主要部分的 **Attn** 从占比和绝对数值上都优化到原本的二分之一。

为了对原始注意力做优化，需要多算一些信息，启动的 Kernel 数都有明显上升，时间上在 Sparge (fp8) 上大幅上升，同时， Sparge (fp8) 的 cuda内核启动时间已经是耗时第二个高 15.7% 的部分了。

## 测试

```bash
python benchmark_end2end.py --compile --attention_type sdpa
# Avg inference time: 88536.48 ms, runs=1
python benchmark_end2end.py --compile --attention_type fa
# Avg inference time: 88432.70 ms, runs=1
python benchmark_end2end.py --compile --attention_type sage2
# Avg inference time: 65574.58 ms, runs=1
python benchmark_end2end.py --compile --attention_type sparge
# Avg inference time: 67230.42 ms, runs=1
python benchmark_end2end.py --compile --attention_type sparge_fp8
# Avg inference time: 64075.88 ms, runs=1


# sta 目前官方只有 Hopper 架构的实现，4090跑不了

nsys profile -t cuda,nvtx -o profile_fa --stats=true --force-overwrite true python benchmark_end2end.py --compile --attention_type fa
nsys profile -t cuda,nvtx -o profile_sage2 --stats=true --force-overwrite true python benchmark_end2end.py --compile --attention_type sage2
nsys profile -t cuda,nvtx -o profile_sparge --stats=true --force-overwrite true python benchmark_end2end.py --compile --attention_type sparge
nsys profile -t cuda,nvtx -o profile_sparge_fp8 --stats=true --force-overwrite true python benchmark_end2end.py --compile --attention_type sparge_fp8
```

## FlashAttn (baseline)

```bash

git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release

# cd hopper  # 支持 hopper 架构
MAX_JOBS=8 python setup.py install # MAX_JOBS 不设小容易并行编译吃完内存被 kill 掉
```

## SageAttn

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
python setup.py install
```

## SpargeAttn

```bash
git clone https://github.com/thu-ml/SpargeAttn.git
cd SpargeAttn
python setup.py install
```
