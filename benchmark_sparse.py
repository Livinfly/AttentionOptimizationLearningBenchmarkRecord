import torch
from torch.nn.functional import scaled_dot_product_attention as default_sdpa # 保存默认的SDPA
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools
import os

from flash_attn.utils.benchmark import benchmark_forward
from spas_sage_attn import spas_sage_attn_meansim_cuda as spargeattn_base_fp16
from spas_sage_attn import spas_sage2_attn_meansim_cuda as spargeattn_base_fp8

parser = argparse.ArgumentParser(description='Benchmark SpargeAttn with varying sparsity parameters at a fixed sequence length.')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--num_heads', type=int, default=32, help='num_heads')
parser.add_argument('--head_dim', type=int, default=128, help='head_dim')
parser.add_argument('--seq_len', type=int, default=32768, help='seq_len')

parser.add_argument('--simthreshd1_values', type=float, nargs='+', default=[0], help='simthreshd1')
parser.add_argument('--cdfthreshd_values', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], help='cdfthreshd')
parser.add_argument('--pvthreshd_values', type=int, nargs='+', default=[50], help='pvthreshd')

parser.add_argument('--output_dir', type=str, default='sparge_sparsity_plots', help='output_dir')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim
seq_len = args.seq_len

print(f"batch={batch_size}, heads={num_heads}, headdim={head_dim}, seq_len={seq_len}")

sparge_functions = {
    "spargeattn_fp16": spargeattn_base_fp16,
    "spargeattn_fp8": spargeattn_base_fp8
}

benchmark_results = []

for is_causal in [False, True]:
    flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len // (2 if is_causal else 1)
    q_global = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k_global = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v_global = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    for func_name, base_fn in sparge_functions.items():
        for sim_val in args.simthreshd1_values:
            for cdf_val in args.cdfthreshd_values:
                for pv_val in args.pvthreshd_values:
                    sparsity_param_config_name = f"sim{sim_val}_cdf{cdf_val}_pv{pv_val}"
                    config_display_name = f"{func_name}_{sparsity_param_config_name}"
                    print(f"\n {config_display_name}")

                    actual_qk_sparsity = float('nan')
                    fn_for_sparsity_check = functools.partial(
                        base_fn,
                        is_causal=is_causal,
                        simthreshd1=sim_val,
                        cdfthreshd=cdf_val,
                        pvthreshd=pv_val,
                        return_sparsity=True
                    )
                    with torch.no_grad():
                        _, actual_qk_sparsity = fn_for_sparsity_check(q_global, k_global, v_global)
                    print(f"    Sparsity {actual_qk_sparsity:.4f}")
                    fn_for_benchmark = functools.partial(
                        base_fn,
                        is_causal=is_causal,
                        simthreshd1=sim_val,
                        cdfthreshd=cdf_val,
                        pvthreshd=pv_val,
                        return_sparsity=False
                    )
                        
                    for _ in range(5):
                        _ = fn_for_benchmark(q_global, k_global, v_global)
                    torch.cuda.synchronize()

                    _, time_obj = benchmark_forward(fn_for_benchmark, q_global, k_global, v_global, 
                                                    is_causal=is_causal,
                                                    repeats=100, 
                                                    verbose=False, desc=config_display_name)
                        
                    mean_time_sec = time_obj.mean
                    mean_time_ms = mean_time_sec * 1000.0
                    
                    tflops_val = (flops / mean_time_sec) * 1e-12
                    
                    print(f"    Time: {mean_time_ms:.3f}ms, TFLOPS: {tflops_val:.3f}")
                    
                    benchmark_results.append({
                        'func_base_name': func_name,
                        'sparsity_param_config': sparsity_param_config_name,
                        'simthreshd1': sim_val,
                        'cdfthreshd': cdf_val,
                        'pvthreshd': pv_val,
                        'actual_qk_sparsity': actual_qk_sparsity,
                        'time_ms': mean_time_ms,
                        'tflops': tflops_val,
                        'seq_len': seq_len,
                        'is_causal': is_causal,
                    })

df_results = pd.DataFrame(benchmark_results)

def plot_sparge_actual_sparsity_charts(df, fixed_seq_len, causal_status_filter, metric_col, y_label, title_prefix, output_dir_path, batch_s, num_h, head_d):
    df_causal_filtered = df[df['is_causal'] == causal_status_filter].copy()
    df_plot = df_causal_filtered.copy()
    df_plot.dropna(subset=['actual_qk_sparsity', metric_col], inplace=True)

    df_plot['qk_sparsity_label'] = (df_plot['actual_qk_sparsity'] * 100).apply(int).astype(str) + '%'
    
    df_plot.sort_values(by=['qk_sparsity_label', 'func_base_name'], inplace=True)
    
    pivot_df = df_plot.pivot_table(index='qk_sparsity_label', columns='func_base_name', values=metric_col, aggfunc='mean')
        
    unique_sorted_sparsity_values = sorted(df_plot['qk_sparsity_label'].unique())
    
    pivot_df = pivot_df.reindex(unique_sorted_sparsity_values)

    num_func_types = len(pivot_df.columns)
    num_x_categories = len(pivot_df.index)
    
    bar_width = 0.8 / num_func_types # 每个条形的宽度
    index_pos = np.arange(num_x_categories) # X轴类别的位置

    fig, ax = plt.subplots(figsize=(max(12, num_x_categories * num_func_types * 0.4), 7)) # 动态调整图表宽度

    color_map = {
        'spargeattn_fp16': 'darkorange',
        'spargeattn_fp8': 'red',
    }

    for i, func_base_name in enumerate(pivot_df.columns):
        offset = (i - (num_func_types - 1) / 2) * bar_width
        
        data_to_plot = pivot_df[func_base_name].fillna(0)
        # print(data_to_plot)
        bars = ax.bar(index_pos + offset, data_to_plot, bar_width, 
                      label=func_base_name, color=color_map.get(func_base_name))
        
        for bar_idx, bar_val in enumerate(data_to_plot):
            # print(bar_val)
            if bar_val > 0 and not np.isnan(bar_val):
                ax.text(index_pos[bar_idx] + offset, bar_val, f'{bar_val:.2f}', 
                         ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Sparsity')
    ax.set_ylabel(y_label)
    ax.set_title(f'{title_prefix}\nSeqLen={fixed_seq_len}, Causal={causal_status_filter}, Batch={batch_s}, Heads={num_h}, Dim={head_d}')
    ax.set_xticks(index_pos)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    ax.legend(title='Sparge type')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir_path, f"{metric_col.lower().replace(' ', '_')}_actual_sparsity_seq{fixed_seq_len}_causal_{str(causal_status_filter).lower()}.png")
    plt.savefig(plot_filename)
    print(f"图表已保存到: {plot_filename}")
    plt.close(fig)

plot_seq_len = df_results['seq_len'].iloc[0] if 'seq_len' in df_results.columns else args.seq_len
    
# 获取其他参数用于标题，假设它们在 args 中
plot_batch_size = args.batch_size
plot_num_heads = args.num_heads
plot_head_dim = args.head_dim


# 为 is_causal = False 生成两张图
plot_sparge_actual_sparsity_charts(df_results, plot_seq_len, False,
                                    metric_col='time_ms', y_label='Time (ms)',
                                    title_prefix='SpargeAttn time vs. Sparsity',
                                    output_dir_path=args.output_dir,
                                    batch_s=plot_batch_size, num_h=plot_num_heads, head_d=plot_head_dim)

plot_sparge_actual_sparsity_charts(df_results, plot_seq_len, False,
                                    metric_col='tflops', y_label='TFLOPS',
                                    title_prefix='SpargeAttn TFLOPS vs. Sparsity',
                                    output_dir_path=args.output_dir,
                                    batch_s=plot_batch_size, num_h=plot_num_heads, head_d=plot_head_dim)

# 为 is_causal = True 生成两张图
plot_sparge_actual_sparsity_charts(df_results, plot_seq_len, True,
                                    metric_col='time_ms', y_label='Time (ms)',
                                    title_prefix='SpargeAttn Time vs. Sparsity',
                                    output_dir_path=args.output_dir,
                                    batch_s=plot_batch_size, num_h=plot_num_heads, head_d=plot_head_dim)

plot_sparge_actual_sparsity_charts(df_results, plot_seq_len, True,
                                    metric_col='tflops', y_label='TFLOPS',
                                    title_prefix='SpargeAttn TFLOPS vs. Sparsity',
                                    output_dir_path=args.output_dir,
                                    batch_s=plot_batch_size, num_h=plot_num_heads, head_d=plot_head_dim)
print(f"\n所有图表已保存在 '{args.output_dir}' 目录中。")

