from torch.nn.functional import scaled_dot_product_attention as sdpa
import torch
from flash_attn.utils.benchmark import benchmark_forward
from sageattention import sageattn
from spas_sage_attn import spas_sage_attn_meansim_cuda as spargeattn
from spas_sage_attn import spas_sage2_attn_meansim_cuda as spargeattn_fp8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools

import argparse

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--output_dir', type=str, default='seq_len_plots', help='Directory to save plots')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
output_dir = args.output_dir

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# spargeattn = functools.partial(
#     spargeattn,
#     simthreshd1=0,
#     cdfthreshd=0.7,
# )
# spargeattn_fp8 = functools.partial(
#     spargeattn_fp8,
#     simthreshd1=0,
#     cdfthreshd=0.7,
# )

print(f"batch: {batch}, head: {head}, headdim: {headdim}")

attention_functions_map = {
    "fa2": sdpa,
    "sage2": sageattn,
    "sparge": spargeattn,
    "sparge_fp8": spargeattn_fp8
}

print(f"Benchmarking functions: {list(attention_functions_map.keys())}")

results_data = []

sequence_lengths = sorted([1024, 2048, 4096, 8192, 16384, 32768])

for is_causal in [False, True]:
    print(f"is_causal: {is_causal}")
    for seq_len in sequence_lengths:
        flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
        q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
        for func_name, fn in attention_functions_map.items():
            for i in range(5): fn(q, k, v, is_causal=is_causal)
            torch.cuda.synchronize()
            _, time = benchmark_forward(fn, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
            print(f'{seq_len} time: {time.mean*1e3}ms flops:{flops/time.mean*1e-12}TOPS, {func_name}')
            results_data.append({
                'func_name': func_name,
                'seq_len': seq_len,
                'time_ms': time.mean*1e3,
                'tflops': flops/time.mean*1e-12,
                'is_causal': is_causal
            })

df_results = pd.DataFrame(results_data)
print(df_results)

def plot_grouped_bar_charts(df, causal_setting, metric_col, y_label, title_suffix, output_dir_path):
    df_filtered = df[df['is_causal'] == causal_setting].copy()
    df_filtered['seq_len_str'] = df_filtered['seq_len'].astype(str)
    pivot_df = df_filtered.pivot_table(index='seq_len_str', columns='func_name', values=metric_col)
    pivot_df = pivot_df.reindex(index=[str(sl) for sl in sequence_lengths if str(sl) in pivot_df.index])

    num_func = len(df_filtered['func_name'].unique())
    num_seq = len(pivot_df.index)
    
    bar_width = 0.8 / num_func
    index = np.arange(num_seq)

    fig, ax = plt.subplots(figsize=(max(12, num_seq * num_func * 0.3), 7))

    for i, func_name in enumerate(pivot_df.columns):
        offset = (i - (num_func - 1) / 2) * bar_width
        bars = ax.bar(index + offset, pivot_df[func_name].fillna(0), bar_width, label=func_name)
        
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                 ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', 
                         ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel(y_label)
    ax.set_title(f'{title_suffix} (is_causal={causal_setting})\nBatch={batch}, Heads={head}, HeadDim={headdim}')
    ax.set_xticks(index)
    ax.set_xticklabels(pivot_df.index)
    ax.legend(title='Attention Function')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir_path, f"{metric_col.lower().replace(' ', '_')}_causal_{str(causal_setting).lower()}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)

if not df_results.empty:
    plot_grouped_bar_charts(df_results, causal_setting=False, metric_col='time_ms', y_label='Time (ms) - Lower is Better', title_suffix='Execution Time', output_dir_path=output_dir)
    plot_grouped_bar_charts(df_results, causal_setting=False, metric_col='tflops', y_label='Performance (TFLOPS) - Higher is Better', title_suffix='Performance', output_dir_path=output_dir)

    plot_grouped_bar_charts(df_results, causal_setting=True, metric_col='time_ms', y_label='Time (ms) - Lower is Better', title_suffix='Execution Time', output_dir_path=output_dir)
    plot_grouped_bar_charts(df_results, causal_setting=True, metric_col='tflops', y_label='Performance (TFLOPS) - Higher is Better', title_suffix='Performance', output_dir_path=output_dir)
