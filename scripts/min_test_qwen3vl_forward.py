#!/usr/bin/env python3
import argparse
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_inputs(processor: AutoProcessor, images, texts, device: torch.device, dtype: torch.dtype):
    # Qwen3-VL expects chat-template text + vision inputs.
    from qwen_vl_utils import process_vision_info

    messages = []
    for img, text in zip(images, texts):
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text},
                    ],
                }
            ]
        )

    prompts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=prompts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    return inputs


def build_img_next_hybrid_4d_mask(
    attention_mask_2d: torch.Tensor,  # [B,T] (1=valid)
    input_ids: torch.Tensor,          # [B,T]
    img_next_id: int,
    block_len: int = 16,
) -> torch.Tensor:
    """
    Additive mask [B,1,T,T]:
      - causal + padding for whole sequence
      - bidirectional inside the last contiguous `block_len` img_next tokens
    """
    if attention_mask_2d.ndim != 2 or input_ids.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {attention_mask_2d.ndim}D/{input_ids.ndim}D")
    if attention_mask_2d.shape != input_ids.shape:
        raise ValueError("attention_mask_2d and input_ids must have same shape")

    B, T = attention_mask_2d.shape
    device = attention_mask_2d.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    min_val = torch.finfo(dtype).min

    valid = attention_mask_2d.to(torch.bool)
    base = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
    allow = base.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)  # [B,T,T]

    img = (input_ids == img_next_id) & valid
    has_img = img.any(dim=1)
    if has_img.any():
        idx = torch.arange(T, device=device)
        end = (img.to(torch.long) * idx).max(dim=1).values
        start = end - (block_len - 1)
        block = (idx.unsqueeze(0) >= start.unsqueeze(1)) & (idx.unsqueeze(0) <= end.unsqueeze(1))
        block_ok = (
            has_img
            & (start >= 0)
            & (block.sum(dim=1) == block_len)
            & ((img & block).sum(dim=1) == block_len)
        )
        if block_ok.any():
            block = block & block_ok.unsqueeze(1)
            allow = allow | (block.unsqueeze(2) & block.unsqueeze(1))

    full = torch.full((B, 1, T, T), min_val, device=device, dtype=dtype)
    full.masked_fill_(allow.unsqueeze(1), 0)
    return full


def _benchmark_forward(
    model: torch.nn.Module,
    inputs: dict,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    dtype: torch.dtype,
):
    # Keep outputs minimal to benchmark attention.
    call_kwargs = dict(output_attentions=False, output_hidden_states=False, return_dict=False, use_cache=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            for _ in range(warmup):
                _ = model(**inputs, **call_kwargs)
            torch.cuda.synchronize()

            starter.record()
            for _ in range(iters):
                _ = model(**inputs, **call_kwargs)
            ender.record()
            torch.cuda.synchronize()

        total_ms = starter.elapsed_time(ender)
    else:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(**inputs, **call_kwargs)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(**inputs, **call_kwargs)
            t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

    return total_ms / max(iters, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="StarVLA/Qwen3-VL-4B-Instruct-Action")
    ap.add_argument("--cache_dir", default="/share/project/lvjing/starVLA/qwen_cache")
    ap.add_argument("--attn_impl", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--try_full_4d_mask", action="store_true")
    ap.add_argument("--append_img_next", action="store_true", help="Append 16 contiguous <img_next> tokens at the end.")
    ap.add_argument("--img_next_token", default="<img_next>")
    ap.add_argument("--use_hybrid_4d_mask", action="store_true", help="Use causal+img_next-bidir 4D mask (SDPA only).")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading model={args.model_id} attn={args.attn_impl} dtype={args.dtype} device={args.device}")
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation=args.attn_impl,
        dtype=dtype,
        device_map="cuda" if device.type == "cuda" else None,
        cache_dir=args.cache_dir,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    t1 = time.perf_counter()
    print(f"Loaded in {t1 - t0:.2f}s")
    print("config._attn_implementation:", getattr(model.config, "_attn_implementation", None))
    print("text_config._attn_implementation:", getattr(model.config.text_config, "_attn_implementation", None))

    img_next_id = None
    if args.append_img_next or args.use_hybrid_4d_mask:
        # Ensure <img_next> is a single token for the tokenizer and model.
        add = processor.tokenizer.add_special_tokens({"additional_special_tokens": [args.img_next_token]})
        if add:
            model.resize_token_embeddings(len(processor.tokenizer))
        img_next_id = processor.tokenizer.convert_tokens_to_ids(args.img_next_token)
        print("img_next:", args.img_next_token, "id=", img_next_id, "added=", add)

    # Dummy images + prompts
    images = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(args.batch)]
    base_text = ("hello " * max(args.words, 1)).strip()
    if args.append_img_next:
        suffix = (" " + args.img_next_token) * 16
    else:
        suffix = ""
    texts = [f"{base_text} {i}{suffix}" for i in range(args.batch)]
    inputs = build_inputs(processor, images, texts, device=device, dtype=dtype)
    if img_next_id is not None and isinstance(inputs.get("input_ids"), torch.Tensor):
        cnt = (inputs["input_ids"] == img_next_id).sum(dim=1).tolist()
        print("img_next_count_per_sample:", cnt)

    # Optional: wrap a fake full 4D mask to see whether it is accepted.
    if args.try_full_4d_mask:
        am = inputs.get("attention_mask")
        print("am.shape:", getattr(am, "shape", None))
        if isinstance(am, torch.Tensor) and am.ndim == 2:
            B, T = am.shape
            full = torch.full((B, 1, T, T), torch.finfo(dtype).min, device=am.device, dtype=dtype)
            allow = torch.tril(torch.ones((T, T), device=am.device, dtype=torch.bool)).unsqueeze(0)
            allow = allow & am.to(torch.bool).unsqueeze(1) & am.to(torch.bool).unsqueeze(2)
            full.masked_fill_(allow.unsqueeze(1), 0)
            print("full.shape:", full.shape)
            # NOTE: Qwen3-VL expects `attention_mask` itself to be a Tensor. A dict will crash inside create_causal_mask.
            inputs["attention_mask"] = full
            print("Using a dummy full 4D mask:", tuple(full.shape))
            if args.attn_impl == "flash_attention_2":
                print("NOTE: flash_attention_2 does not support dense 4D masks; switching config to sdpa for this test.")
                model.config._attn_implementation = "sdpa"
                model.config.text_config._attn_implementation = "sdpa"

    if args.use_hybrid_4d_mask:
        if args.attn_impl != "sdpa":
            raise ValueError("--use_hybrid_4d_mask requires --attn_impl sdpa")
        if img_next_id is None:
            raise ValueError("img_next_id is None (unexpected)")
        am2d = inputs.get("attention_mask")
        ids = inputs.get("input_ids")
        if not (isinstance(am2d, torch.Tensor) and isinstance(ids, torch.Tensor) and am2d.ndim == 2 and ids.ndim == 2):
            raise ValueError("Expected 2D attention_mask + input_ids from processor")
        full = build_img_next_hybrid_4d_mask(am2d, ids, img_next_id=img_next_id, block_len=16)
        inputs["attention_mask"] = full
        print("Using hybrid 4D mask:", tuple(full.shape))

    avg_ms = _benchmark_forward(
        model,
        inputs,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        dtype=dtype,
    )
    print(f"bench: batch={args.batch} words={args.words} warmup={args.warmup} iters={args.iters} avg_ms={avg_ms:.3f}")


if __name__ == "__main__":
    main()
