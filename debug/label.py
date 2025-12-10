import logging
import os
import sys
import warnings

from tqdm import tqdm

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from omegaconf import OmegaConf

from utils.dataset import get_dataloader
from utils.logger import HTRLogger


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


if __name__ == "__main__":
    config = parse_args()

    max_epochs = config.train.num_epochs
    device = config.device if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    image_ext = getattr(config.data, "image_ext", ".png")
    dummy_loader = get_dataloader(
        config.data.dummy_images_dir,
        config.data.dummy_labels_path,
        processor,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.eval.num_workers,
        image_ext=image_ext,
    )

    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)


    model.config.pad_token_id = 1
    model.config.eos_token_id = 2
    model.config.decoder_start_token_id = 2
    model = model.to(device)

    model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )


    for epoch in range(1, max_epochs + 1):
        pbar = tqdm(
            enumerate(dummy_loader, 1),
            total=len(dummy_loader),
            desc=f"Epoch {epoch}",
        )
        optimizer.zero_grad()
        loss_val = 0.0
        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            print(f"labels device: {labels.device}")
            # ラベルが正しいか確認用コマンド
            print("☆"*100)
            print(f"labels sample: {labels.shape}")
            labels_cpu = labels.cpu(); 
            decoded = processor.batch_decode(labels_cpu.masked_fill(labels_cpu == -100, processor.tokenizer.pad_token_id),skip_special_tokens=False)
            print(decoded[0])
            print(decoded.shape)
            
        break
            