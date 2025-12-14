from typing import List, Dict, Any
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import functools


def _identity(x):
    """Default text normalizer that returns the input unchanged."""
    return x


class IAMDataset(Dataset):
    """
    Simple OCR dataset for (image, text) pairs.

    labels.txt format:
        <image_id> <text>
        e.g.) img001 Hello World

    images are expected at: <images_dir>/<image_id><image_ext>
    """

    def __init__(self, images_dir: str, labels_txt: str, normalize_fn=None, image_ext: str = ".png", transform=None):
        self.images_dir = images_dir
        self.normalize_fn = normalize_fn if normalize_fn is not None else _identity
        self.image_ext = image_ext
        self.transform = transform
        self.samples = []  # list of (image_path, text, image_id)

        with open(labels_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 0:
                    continue
                image_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                img_path = os.path.join(images_dir, image_id + image_ext)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.normalize_fn(text), image_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, text, image_id = self.samples[idx]
        image = load_rgb(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "text": text, "id": image_id, "image_path": img_path}


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    return img


def collate_fn(batch: List[Dict[str, Any]], processor, max_target_length: int = 128) -> Dict[str, Any]:
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    ids = [x["id"] for x in batch]

    encodings = processor(images=images, return_tensors="pt")
    pixel_values = encodings.pixel_values

    tokenizer = processor.tokenizer
    labels = tokenizer(
        text_target=texts,
        padding="longest",
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    ).input_ids

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels[labels == pad_id] = -100

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "ids": ids,
    }


def get_dataloader(
    images_dir: str,
    labels_txt: str,
    processor,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_target_length: int = 128,
    image_ext: str = ".png",
    transform=None,
) -> DataLoader:
    ds = IAMDataset(images_dir, labels_txt, normalize_fn=None, image_ext=image_ext, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_fn, processor=processor, max_target_length=max_target_length),
    )
