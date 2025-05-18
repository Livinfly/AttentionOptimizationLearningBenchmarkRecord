# Benchmark_EXP

默认以 LLaMa3 为例，测试各个模块的加速情况。

## FlashAttn (baseline)

```bash
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
python setup.py install
```

## FlashInfer

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install -e . -v
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

## STA (fastvideo)

```bash
pip install fastvideo
```

