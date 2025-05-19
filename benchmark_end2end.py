import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX-2b", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'fa', 'sage2', 'sparge', 'sparge_fp8', 'flashinfer', 'sta'], help='Attention type')
parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
parser.add_argument('--runs', type=int, default=1, help='Number of measuremd runs')
args = parser.parse_args()

if args.attention_type == 'sage2':
    from sageattention import sageattn
    F.scaled_dot_product_attention = sageattn
elif args.attention_type == 'fa':
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
elif args.attention_type == 'sparge':
    from spas_sage_attn import spas_sage_attn_meansim_cuda as sparge_attn
    F.scaled_dot_product_attention = sparge_attn
elif args.attention_type == 'sparge_fp8':
    from spas_sage_attn import spas_sage2_attn_meansim_cuda as sparge_attn
    F.scaled_dot_product_attention = sparge_attn
# elif args.attention_type == 'flashinfer':
#     from wrapper import fi
#     F.scaled_dot_product_attention = fi
# elif args.attention_type == 'sta':
#     from fastvideo import VideoGenerator
#     F.scaled_dot_product_attention = sta_attn

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16
).to("cuda")

if args.compile:
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Warmup runs
print(f"Starting {args.warmup} warmup runs ...")
with torch.no_grad():
    for _ in range(args.warmup):
        _ = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=10,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

# Measured runs
print(f"Starting {args.runs} measured runs ...")
timings = []
with torch.no_grad():
    for i in range(args.runs):
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        timings.append(elapsed)
        print(f"Run {i+1} inference time: {elapsed:.2f} ms")

mean_time = sum(timings) / len(timings)
print(f"Avg inference time: {mean_time:.2f} ms, runs={args.runs}")

# export_to_video(video, f"cogvideox-2b_{args.attention_type}.mp4", fps=8)