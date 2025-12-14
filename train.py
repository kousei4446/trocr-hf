import logging
import os
import sys
import warnings

from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from omegaconf import OmegaConf

from utils.dataset import get_dataloader
from utils.metrics import CER, normalize_36_charset
from utils.logger import HTRLogger
from utils.transforms import build_train_transform

# Transformers/tokenizers の冗長ログ抑制
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*as_target_tokenizer.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*Some weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*Config of the.*is overwritten.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


def evaluate(model, processor, loader, device, id2gt, desc=None, max_new_tokens=64):
    model.eval()
    cer_metric = CER(normalize_fn=lambda t: normalize_36_charset(t, keep_space=False))
    iterator = tqdm(loader, desc=desc) if desc else loader

    with torch.no_grad():
        for batch in iterator:
            pixel_values = batch["pixel_values"].to(device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": model.config.decoder_start_token_id,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            gen_out = model.generate(pixel_values, **gen_kwargs)
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            sequences = sequences.detach().cpu()
            preds = processor.batch_decode(sequences, skip_special_tokens=True)

            gts = [id2gt.get(img_id, "") for img_id in batch["ids"]]

            for p, g in zip(preds, gts):
                cer_metric.update(p, g)

    model.train()
    return cer_metric.score()


def save_model_and_processor(model, processor, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, f"epoch-{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    elif hasattr(model, "encoder_decoder"):
        model.encoder_decoder.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "epoch": epoch},
        os.path.join(ckpt_dir, "optim.pt"),
        _use_new_zipfile_serialization=False,
    )
    print(f"Saved checkpoint to {ckpt_dir}")


def _default_model_builder(config):
    # 遅延 import で循環参照を避ける
    from models.model import build_model

    return build_model(config)


def _compute_loss(outputs, labels):
    logits = outputs.logits
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )


def run_training(config, build_model_fn=None):
    max_epochs = config.train.num_epochs
    device = config.device if torch.cuda.is_available() else "cpu"

    model_builder = build_model_fn or _default_model_builder
    model = model_builder(config)

    processor = getattr(model, "processor", None)
    if processor is None:
        processor = TrOCRProcessor.from_pretrained(config.model_name)

    train_transform = None
    if config.data.augmentations.enable:
        train_transform = build_train_transform(config)

    image_ext = getattr(config.data, "image_ext", ".png")
    train_loader = get_dataloader(
        config.data.train_images_dir,
        config.data.train_labels_path,
        processor,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        image_ext=image_ext,
        transform=train_transform,
    )
    val_loader = get_dataloader(
        config.data.val_images_dir,
        config.data.val_labels_path,
        processor,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.eval.num_workers,
        image_ext=image_ext,
    )
    test_loader = get_dataloader(
        config.data.test_images_dir,
        config.data.test_labels_path,
        processor,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.eval.num_workers,
        image_ext=image_ext,
    )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    id2gt_test = {iid: txt for (_p, txt, iid) in test_loader.dataset.samples}
    logger = HTRLogger(log_dir=config.logging.log_dir, config=config)

    for epoch in range(1, max_epochs + 1):
        pbar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
        )
        optimizer.zero_grad()
        loss_val = 0.0

        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            main_loss = _compute_loss(outputs, labels)
            llm_loss_val = getattr(outputs, "llm_loss", None)
            llm_loss_layers = getattr(outputs, "llm_loss_per_layer", None)

            total_loss = main_loss
            if llm_loss_val is not None:
                total_loss = total_loss + config.train.llm_loss_weight * llm_loss_val

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_val += total_loss.item()
            pbar.set_postfix({"loss": f"{(loss_val / step):.4f}"})
            logger.log_step(
                total_loss.item(),
                epoch,
                step,
                main_loss=main_loss.item(),
                llm_loss=llm_loss_val.item() if llm_loss_val is not None else None,
                llm_loss_layers={k: v.item() for k, v in llm_loss_layers.items()} if llm_loss_layers else None,
            )

        if epoch % config.train.save_interval == 0:
            save_model_and_processor(model, processor, optimizer, epoch, config.model.save_dir)

        # if epoch % config.train.eval_interval == 0:
        cer_test = evaluate(
            model,
            processor,
            test_loader,
            device,
            id2gt_test,
            desc=f"Test epoch {epoch}",
        )
        print(f"Test CER: {cer_test:.4f}")
        logger.log_test(epoch, cer_test)

    logger.close()


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
