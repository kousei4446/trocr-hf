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
from utils.augmentations import (
    Compose,
    RandomRotate,
    RandomGaussianBlur,
    RandomDilation,
    RandomErosion,
    RandomDownsample,
    RandomUnderline,
)


def build_train_transform(config):
    aug_cfg = getattr(config.data, "augmentations", None)
    if not aug_cfg or not getattr(aug_cfg, "enable", False):
        return None

    resample_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }

    transforms = []

    rotate_cfg = getattr(aug_cfg, "rotate", None)
    if rotate_cfg:
        transforms.append(
            RandomRotate(
                degrees=getattr(rotate_cfg, "degrees", 10.0),
                p=getattr(rotate_cfg, "p", 0.5),
                fill=getattr(rotate_cfg, "fill", 255),
            )
        )

    blur_cfg = getattr(aug_cfg, "gaussian_blur", None)
    if blur_cfg:
        transforms.append(
            RandomGaussianBlur(
                radius_range=(
                    getattr(blur_cfg, "radius_min", 0.5),
                    getattr(blur_cfg, "radius_max", 1.2),
                ),
                p=getattr(blur_cfg, "p", 0.3),
            )
        )

    dil_cfg = getattr(aug_cfg, "dilation", None)
    if dil_cfg:
        transforms.append(
            RandomDilation(
                size=getattr(dil_cfg, "size", 3),
                p=getattr(dil_cfg, "p", 0.3),
            )
        )

    ero_cfg = getattr(aug_cfg, "erosion", None)
    if ero_cfg:
        transforms.append(
            RandomErosion(
                size=getattr(ero_cfg, "size", 3),
                p=getattr(ero_cfg, "p", 0.3),
            )
        )

    down_cfg = getattr(aug_cfg, "downsample", None)
    if down_cfg:
        up_name = getattr(down_cfg, "up_interpolation", "bicubic")
        up_mode = resample_map.get(str(up_name).lower(), Image.BICUBIC)
        transforms.append(
            RandomDownsample(
                ratio_range=(
                    getattr(down_cfg, "ratio_min", 0.5),
                    getattr(down_cfg, "ratio_max", 0.8),
                ),
                p=getattr(down_cfg, "p", 0.4),
                up_interpolation=up_mode,
            )
        )

    ul_cfg = getattr(aug_cfg, "underline", None)
    if ul_cfg:
        transforms.append(
            RandomUnderline(
                p=getattr(ul_cfg, "p", 0.3),
                thickness_range=(
                    getattr(ul_cfg, "thickness_min", 1),
                    getattr(ul_cfg, "thickness_max", 3),
                ),
                offset_range=(
                    getattr(ul_cfg, "offset_min", 1),
                    getattr(ul_cfg, "offset_max", 5),
                ),
                color=tuple(getattr(ul_cfg, "color", (0, 0, 0))),
            )
        )

    return Compose(transforms) if transforms else None


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    # allow optional dummy_* keys
    OmegaConf.set_struct(conf, False)
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

    # use dummy if provided, otherwise fall back to train
    images_dir = config.data.get("dummy_images_dir", config.data.train_images_dir)
    labels_path = config.data.get("dummy_labels_path", config.data.train_labels_path)

    # デフォルト拡張子は .png。dummy が .jpg の場合は config.data.image_ext で上書き可能。
    image_ext = config.data.get("image_ext", ".png")

    train_transform = build_train_transform(config)

    train_loader = get_dataloader(
        images_dir,
        labels_path,
        processor,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        image_ext=image_ext,
        transform=train_transform,
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
