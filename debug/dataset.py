"""
Quick dataloader/debug script.
Run: python -m debug.dataset config.yaml
"""
import os
import sys
from typing import List

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor

from utils.dataset import get_dataloader


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # remove config file name from argv
    conf.merge_with_cli()
    return conf


def to_pil_batch(pixel_values) -> List[Image.Image]:
    """
    Convert normalized pixel_values (B,C,H,W) back to PIL images for inspection.
    Assumes ImageNet normalization mean/std used by TrOCR.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imgs = []
    pv_cpu = pixel_values.detach().cpu()
    for pv in pv_cpu:  # (C,H,W)
        img = pv.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        imgs.append(Image.fromarray((img * 255).astype(np.uint8)))
    return imgs


if __name__ == "__main__":
    config = parse_args()
    os.makedirs(config.debug.save_dir, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    train_loader = get_dataloader(
        config.data.dummy_images_dir,
        config.data.dummy_labels_path,
        processor,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )

    print(f"Dataloader ready: {len(train_loader.dataset)} samples, batch_size={config.train.batch_size}")

    for epoch in range(1, 2):  # one pass is enough for debugging
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            ids = batch.get("ids", [])

            print("\n--- Batch preview ---")
            print(f"pixel_values shape: {tuple(pixel_values.shape)}")
            print(f"labels shape: {tuple(labels.shape)}")

            # Decode labels to text for a quick sanity check
            labels_for_decode = labels.detach().cpu().clone()
            pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            labels_for_decode[labels_for_decode == -100] = pad_id
            decoded = processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            for i, txt in enumerate(decoded[:3]):
                sample_id = ids[i] if i < len(ids) else f"{step}-{i}"
                print(f"[{sample_id}] {txt}")

            # Save images for visual inspection
            imgs = to_pil_batch(pixel_values)
            for i, img in enumerate(imgs):
                sample_id = ids[i] if i < len(ids) else f"{step}-{i}"
                fname = f"ep{epoch:02d}_st{step:04d}_{i:03d}_{sample_id}.png"
                img.save(os.path.join(config.debug.save_dir, fname))

            pbar.set_postfix({"saved_imgs": len(imgs)})
            # Only first batch is sufficient for debugging
            break
