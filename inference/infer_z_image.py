#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Z-Image LoRA Inference Script (Standalone)

Self-contained inference script for Z-Image with LoRA.
Reads captions from metadata.jsonl and loads LoRA from the repo directory.

Requirements:
    pip install torch diffusers peft accelerate transformers pillow tqdm

Usage:
  # Single GPU
  python inference/infer_z_image.py \
    --base_model_path Tongyi-MAI/Z-Image \
    --lora_path ./z-image-prompt_echo_lora \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_z_image

  # Multi-GPU
  accelerate launch --num_processes 8 inference/infer_z_image.py \
    --base_model_path Tongyi-MAI/Z-Image \
    --lora_path ./z-image-prompt_echo_lora \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_z_image

  # Base model only (no LoRA)
  python inference/infer_z_image.py \
    --base_model_path Tongyi-MAI/Z-Image \
    --caption_jsonl ./metadata.jsonl \
    --output_dir ./output_z_image_base
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
# Inline pipeline_simple (from z_image_pipeline_simple.py)
# ============================================================

def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


@torch.no_grad()
def pipeline_simple(
    transformer,
    vae,
    scheduler,
    prompt_embeds: torch.FloatTensor,
    attention_mask: torch.BoolTensor,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 4.0,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_attention_mask: Optional[torch.BoolTensor] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Simplified Z-Image pipeline with classifier-free guidance.

    Returns:
        images: (B, C, H, W) in [0, 1]
    """
    if device is None:
        device = prompt_embeds.device
    if dtype is None:
        dtype = prompt_embeds.dtype

    batch_size = prompt_embeds.shape[0]
    do_cfg = guidance_scale > 1.0

    # Convert padded prompt_embeds to List[Tensor] (transformer expects this)
    prompt_embeds_list = []
    for i in range(batch_size):
        prompt_embeds_list.append(prompt_embeds[i][attention_mask[i]])

    negative_prompt_embeds_list = []
    if do_cfg:
        if negative_prompt_embeds is None or negative_attention_mask is None:
            raise ValueError("negative_prompt_embeds required when guidance_scale > 1.0")
        for i in range(batch_size):
            negative_prompt_embeds_list.append(
                negative_prompt_embeds[i][negative_attention_mask[i]]
            )

    # Prepare latents (4D format: B, C, H, W)
    vae_scale_factor = 8
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    latent_channels = 16

    if latents is None:
        latents = torch.randn(
            (batch_size, latent_channels, latent_h, latent_w),
            device=device, dtype=torch.float32,
            generator=generator[0] if isinstance(generator, list) else generator,
        )

    # Timesteps with shift
    image_seq_len = (latent_h // 2) * (latent_w // 2)
    mu = _calculate_shift(image_seq_len)
    timesteps, num_inference_steps = _retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
    )

    # Denoising loop
    for i, t in enumerate(timesteps):
        timestep = t.expand(batch_size)
        timestep_norm = (1000 - timestep) / 1000

        if do_cfg:
            latent_model_input = latents.to(dtype).repeat(2, 1, 1, 1)
            timestep_model_input = timestep_norm.repeat(2)
            prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
        else:
            latent_model_input = latents.to(dtype)
            timestep_model_input = timestep_norm
            prompt_embeds_model_input = prompt_embeds_list

        # Add frame dimension (Z-Image expects [B, C, 1, H, W])
        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        model_out_list = transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            return_dict=False,
        )[0]

        if do_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            noise_pred = []
            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                noise_pred.append(pos + guidance_scale * (pos - neg))
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

        # Remove frame dim and negate (Z-Image specific)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        latents = scheduler.step(
            noise_pred.to(torch.float32), t, latents, return_dict=False,
        )[0]

    # VAE decode
    latents_for_decode = latents.to(vae.dtype)
    latents_for_decode = (latents_for_decode / vae.config.scaling_factor) + vae.config.shift_factor
    images = vae.decode(latents_for_decode, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)

    return images


# ============================================================
# Text encoding (Z-Image uses Qwen tokenizer with chat template)
# ============================================================

def compute_text_embeddings(prompts, text_encoder, tokenizer, max_length, device, dtype):
    """Encode prompts for Z-Image using chat template formatting."""
    with torch.no_grad():
        formatted = list(prompts)
        for i, prompt in enumerate(formatted):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted[i] = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                )
            except TypeError:
                formatted[i] = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

        inputs = tokenizer(
            formatted, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device).bool()

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-2].to(dtype)

    return prompt_embeds, attention_mask


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
    parser = argparse.ArgumentParser(description="Z-Image LoRA Inference")
    parser.add_argument("--base_model_path", type=str, default="Tongyi-MAI/Z-Image",
                        help="HuggingFace model ID or local path for Z-Image base model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--caption_jsonl", type=str, default="./metadata.jsonl",
                        help="Caption JSONL file (each line: {\"caption\": \"...\"})")
    parser.add_argument("--output_dir", type=str, default="./output_z_image",
                        help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of images to generate (default: all captions)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (height=width)")
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="CFG guidance scale")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-GPU batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_words", type=int, default=200,
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
        print(f"=== Z-Image Inference ===")
        print(f"  Base model: {args.base_model_path}")
        print(f"  LoRA: {args.lora_path or '(none - base model)'}")
        print(f"  Resolution: {args.resolution}x{args.resolution}")
        print(f"  Steps: {args.num_steps}, CFG: {args.guidance_scale}")
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

    # --- Load model components ---
    if is_main:
        print(f"Loading Z-Image from: {args.base_model_path}")
    load_start = time.time()

    from diffusers import AutoencoderKL
    from diffusers.models import ZImageTransformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer

    inference_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(
        args.base_model_path, subfolder="vae", torch_dtype=inference_dtype,
    )
    vae.requires_grad_(False)
    vae.eval()

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.base_model_path, subfolder="transformer", torch_dtype=inference_dtype,
    )

    text_encoder = AutoModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder",
        torch_dtype=inference_dtype, trust_remote_code=True,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, subfolder="tokenizer", trust_remote_code=True,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler",
    )

    if is_main:
        print(f"Model loaded in {time.time() - load_start:.1f}s")

    # --- Load LoRA (optional) ---
    if args.lora_path:
        if is_main:
            print(f"Loading LoRA from: {args.lora_path}")

        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, args.lora_path)
        transformer.set_adapter("default")
        transformer = transformer.merge_and_unload()

        if is_main:
            print("LoRA merged successfully")

    transformer.requires_grad_(False)
    transformer.eval()

    # --- Move to GPU ---
    vae.to(device, dtype=torch.float32)  # VAE needs float32
    text_encoder.to(device, dtype=inference_dtype)
    transformer.to(device, dtype=inference_dtype)

    # --- Pre-compute empty prompt embeddings for CFG ---
    empty_prompt_embeds, empty_attention_mask = compute_text_embeddings(
        [""], text_encoder, tokenizer, max_length=512, device=device, dtype=inference_dtype,
    )
    if is_main:
        print(f"Empty prompt embeddings: {empty_prompt_embeds.shape}")

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

        # Encode prompts
        prompt_embeds, attention_mask = compute_text_embeddings(
            prompts, text_encoder, tokenizer,
            max_length=512, device=device, dtype=inference_dtype,
        )

        neg_embeds = empty_prompt_embeds.expand(batch_size, -1, -1)
        neg_mask = empty_attention_mask.expand(batch_size, -1)

        # Generate
        scheduler_copy = copy.deepcopy(scheduler)
        with torch.cuda.amp.autocast(dtype=inference_dtype):
            with torch.no_grad():
                images = pipeline_simple(
                    transformer, vae, scheduler_copy,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=neg_embeds,
                    negative_attention_mask=neg_mask,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    dtype=inference_dtype,
                )

        # Save images
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
