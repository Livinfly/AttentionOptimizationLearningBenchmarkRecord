# Benchmark_EXP

默认以 CogVideoX-2b 为例，测试各个模块的加速情况。
由于使用 RTX 4090 测试，不是 Hopper 架构，不能选择 fa3, sta 测试。
测试对象为：sage2, sparge, sparge_fp8
baseline: torch backend flash_sdp

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

nsys profile -t cuda,nvtx -o profile_fa --stats=true --force-overwrite true python benchmark_end2end.py --comile --attnetion_type fa
nsys profile -t cuda,nvtx -o profile_sage2 --stats=true --force-overwrite true python benchmark_end2end.py --comile --attnetion_type sage2
nsys profile -t cuda,nvtx -o profile_sparge --stats=true --force-overwrite true python benchmark_end2end.py --comile --attnetion_type sparge
nsys profile -t cuda,nvtx -o profile_sparge_fp8 --stats=true --force-overwrite true python benchmark_end2end.py --comile --attnetion_type sparge_fp8
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

