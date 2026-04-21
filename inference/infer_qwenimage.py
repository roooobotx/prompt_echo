#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen-Image LoRA Inference Script (Standalone)

Self-contained inference script for Qwen-Image with LoRA.
Reads captions from metadata.jsonl and loads LoRA from the repo directory.

Requirements:
    pip install torch diffusers peft accelerate transformers pillow tqdm

Usage:
  # Single GPU
  python inference/infer_qwenimage.py \
    --base_model_path Qwen/Qwen-Image-2512 \
    --lora_path ./qwenimage-prompt_echo_lora \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_qwenimage

  # Multi-GPU
  accelerate launch --num_processes 8 inference/infer_qwenimage.py \
    --base_model_path Qwen/Qwen-Image-2512 \
    --lora_path ./qwenimage-prompt_echo_lora \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_qwenimage

  # Base model only (no LoRA)
  python inference/infer_qwenimage.py \
    --base_model_path Qwen/Qwen-Image-2512 \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_qwenimage_base
"""

import argparse
import copy
import json
import os
import time
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# Inline pipeline_simple (from qwenimage_pipeline_simple.py)
# ============================================================

from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps


@torch.no_grad()
def pipeline_simple(
    pipeline,
    prompts: List[str],
    negative_prompts: List[str],
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    sigmas: Optional[List[float]] = None,
    true_cfg_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 512,
):
    """
    Simplified Qwen-Image pipeline with norm-guided CFG.

    Returns:
        images: (B, C, H, W) in [0, 1]
    """
    device = pipeline._execution_device
    batch_size = len(prompts)

    transformer = pipeline.transformer
    transformer_unwrapped = transformer.module if hasattr(transformer, 'module') else transformer

    # 1. Encode prompts (positive + negative together, then split)
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        prompt=prompts + negative_prompts,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    seq_len = prompt_embeds.shape[1]
    if seq_len < max_sequence_length:
        pad_len = max_sequence_length - seq_len
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_len), value=0)
        prompt_embeds_mask = torch.nn.functional.pad(prompt_embeds_mask, (0, pad_len), value=0)
    prompt_embeds, negative_prompt_embeds = prompt_embeds.chunk(2, dim=0)
    prompt_embeds_mask, negative_prompt_embeds_mask = prompt_embeds_mask.chunk(2, dim=0)

    # 2. Prepare latents (3D packed format)
    num_channels_latents = transformer_unwrapped.config.in_channels // 4
    latents = pipeline.prepare_latents(
        batch_size, num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents,
    )

    # 3. Auxiliary inputs
    img_shapes = [[(1, height // pipeline.vae_scale_factor // 2,
                     width // pipeline.vae_scale_factor // 2)]] * batch_size
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
    negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()

    if transformer_unwrapped.config.guidance_embeds:
        guidance = torch.full([1], true_cfg_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # 4. Timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
    )

    # 5. Denoising loop (ODE with norm-guided CFG)
    pipeline.scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = pipeline.transformer(
            hidden_states=torch.cat([latents, latents], dim=0),
            timestep=torch.cat([timestep, timestep], dim=0) / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0),
            encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
            img_shapes=img_shapes * 2,
            txt_seq_lens=txt_seq_lens + negative_txt_seq_lens,
        )[0]

        # Norm-guided CFG
        noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)

        latents_dtype = latents.dtype
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    # 6. VAE decode
    latents_for_decode = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents_for_decode = latents_for_decode.to(pipeline.vae.dtype)
    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(latents_for_decode.device, latents_for_decode.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(pipeline.vae.config.latents_std)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(latents_for_decode.device, latents_for_decode.dtype)
    )
    latents_for_decode = latents_for_decode / latents_std + latents_mean
    image = pipeline.vae.decode(latents_for_decode, return_dict=False)[0][:, :, 0]
    image = pipeline.image_processor.postprocess(image, output_type="pt")

    return image


# ============================================================
# Caption dataset (inline, reads from metadata.jsonl)
# ============================================================

class CaptionDataset(torch.utils.data.Dataset):
    """Simple caption dataset that reads from metadata.jsonl (only 'caption' field)."""

    def __init__(self, jsonl_path, max_samples=None, max_words=None, seed=42):
        self.captions = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.captions.append(data.get("caption", ""))

        if seed is not None:
            import random
            random.seed(seed)
            random.shuffle(self.captions)

        if max_samples is not None:
            self.captions = self.captions[:max_samples]

        self.max_words = max_words

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        if self.max_words is not None:
            words = caption.split()
            if len(words) > self.max_words:
                caption = " ".join(words[:self.max_words])
        return {"prompt": caption, "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        indices = [e["index"] for e in examples]
        return prompts, indices


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image LoRA Inference")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen-Image-2512",
                        help="HuggingFace model ID or local path for base model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--caption_jsonl", type=str, default="./metadata.jsonl",
                        help="Caption JSONL file (each line: {\"caption\": \"...\"})")
    parser.add_argument("--output_dir", type=str, default="./output_qwenimage",
                        help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of images to generate (default: all captions)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (height=width)")
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="True CFG scale (norm-guided)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_words", type=int, default=500,
                        help="Max words per caption")
    parser.add_argument("--image_format", type=str, default="jpg",
                        choices=["jpg", "png"], help="Output image format")
    return parser.parse_args()


def main():
    args = parse_args()

    # Try to use accelerate for multi-GPU, fall back to single GPU
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from torch.utils.data.distributed import DistributedSampler

        accelerator = Accelerator(mixed_precision="bf16")
        rank = accelerator.process_index
        num_processes = accelerator.num_processes
        device = accelerator.device
        is_main = accelerator.is_main_process
        set_seed(args.seed)
        multi_gpu = num_processes > 1
    except ImportError:
        rank = 0
        num_processes = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        multi_gpu = False
        torch.manual_seed(args.seed)

    if is_main:
        print(f"=== Qwen-Image Inference ===")
        print(f"  Base model: {args.base_model_path}")
        print(f"  LoRA: {args.lora_path or '(none - base model)'}")
        print(f"  Resolution: {args.resolution}x{args.resolution}")
        print(f"  Steps: {args.num_steps}, True CFG: {args.true_cfg_scale}")
        print(f"  GPUs: {num_processes}")

    # --- Dataset ---
    dataset = CaptionDataset(
        args.caption_jsonl,
        max_samples=args.num_samples,
        max_words=args.max_words,
        seed=args.seed,
    )

    if multi_gpu:
        sampler = DistributedSampler(dataset, num_replicas=num_processes, rank=rank, shuffle=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=CaptionDataset.collate_fn, num_workers=2, pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=CaptionDataset.collate_fn, num_workers=2, pin_memory=True,
        )

    if is_main:
        print(f"  Captions: {len(dataset)}")

    # --- Load pipeline ---
    if is_main:
        print(f"Loading pipeline from: {args.base_model_path}")
    load_start = time.time()

    from diffusers import DiffusionPipeline
    inference_dtype = torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(
        args.base_model_path, torch_dtype=inference_dtype,
    )
    pipeline.safety_checker = None

    if is_main:
        print(f"Pipeline loaded in {time.time() - load_start:.1f}s")

    # --- Load LoRA (optional) ---
    if args.lora_path:
        if is_main:
            print(f"Loading LoRA from: {args.lora_path}")

        # Check if it's a peft-format directory (has adapter_config.json)
        if os.path.isfile(os.path.join(args.lora_path, "adapter_config.json")):
            from peft import PeftModel
            transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_path)
            transformer.set_adapter("default")
            transformer = transformer.merge_and_unload()
            pipeline.transformer = transformer
        elif os.path.isfile(os.path.join(args.lora_path, "pytorch_lora_weights.safetensors")):
            pipeline.load_lora_weights(args.lora_path)
            pipeline.fuse_lora()
            pipeline.unload_lora_weights()
        else:
            raise FileNotFoundError(
                f"No LoRA weights found in {args.lora_path}. "
                f"Expected adapter_config.json (peft) or pytorch_lora_weights.safetensors (diffusers)."
            )
        if is_main:
            print("LoRA merged successfully")

    pipeline.transformer.requires_grad_(False)
    pipeline.transformer.eval()

    # --- Move to GPU ---
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.transformer.to(device, dtype=inference_dtype)

    # --- Output directory ---
    images_dir = os.path.join(args.output_dir, "images")
    if is_main:
        os.makedirs(images_dir, exist_ok=True)
    if multi_gpu:
        accelerator.wait_for_everyone()
    os.makedirs(images_dir, exist_ok=True)

    # --- Inference loop ---
    metadata_entries = []
    total_generated = 0

    for batch_idx, (prompts, indices) in enumerate(tqdm(
        dataloader,
        desc=f"[GPU {rank}] Generating",
        disable=not is_main,
    )):
        batch_size = len(prompts)

        with torch.cuda.amp.autocast(dtype=inference_dtype):
            with torch.no_grad():
                images = pipeline_simple(
                    pipeline, prompts,
                    negative_prompts=[" "] * batch_size,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_steps,
                    true_cfg_scale=args.true_cfg_scale,
                    max_sequence_length=512,
                )

        images_np = images.cpu().float().numpy()
        for i in range(batch_size):
            sample_idx = indices[i]
            filename = f"{sample_idx:04d}.{args.image_format}"
            filepath = os.path.join(images_dir, filename)

            img_np = (images_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(filepath)

            metadata_entries.append({
                "index": sample_idx,
                "filename": f"images/{filename}",
                "caption": prompts[i],
            })

        total_generated += batch_size

    print(f"[GPU {rank}] Generated {total_generated} images")

    # --- Save metadata ---
    if multi_gpu:
        rank_path = os.path.join(args.output_dir, f"metadata_rank{rank}.jsonl")
        with open(rank_path, "w", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        accelerator.wait_for_everyone()

        if is_main:
            all_entries = []
            for r in range(num_processes):
                rpath = os.path.join(args.output_dir, f"metadata_rank{r}.jsonl")
                if os.path.exists(rpath):
                    with open(rpath, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                all_entries.append(json.loads(line))
            all_entries.sort(key=lambda x: x["index"])
            merged_path = os.path.join(args.output_dir, "metadata.jsonl")
            with open(merged_path, "w", encoding="utf-8") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for r in range(num_processes):
                rpath = os.path.join(args.output_dir, f"metadata_rank{r}.jsonl")
                if os.path.exists(rpath):
                    os.remove(rpath)
    else:
        merged_path = os.path.join(args.output_dir, "metadata.jsonl")
        with open(merged_path, "w", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if is_main:
        print(f"\n=== Done === {total_generated} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
