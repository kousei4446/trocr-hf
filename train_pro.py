# -*- coding: utf-8 -*-
"""
train_trocr_ft.py - TrOCR fine-tuning (encoder も学習する改良版)
"""

import warnings
import logging
import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    get_linear_schedule_with_warmup,
)
from omegaconf import OmegaConf

from utils.dataset import get_dataloader
from utils.metrics import CER, WER
from utils.logger import HTRLogger


# ===== ログ・警告抑制 =====
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


# ===== 設定読み込み =====
def parse_args():
    """
    使い方:
        python train_trocr_ft.py config.yaml
    """
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    # hydra 風 override のために、config ファイル名を argv から外す
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


# ===== id -> GT 文字列 辞書 =====
def build_id2gt(dataset):
    """
    dataset.samples: List[(img_path, text, image_id)]
    """
    return {iid: txt for (pth, txt, iid) in dataset.samples}


# ===== 評価ループ（val / test 共通） =====
def run_eval(model, processor, loader, device, id2gt, wer_mode):
    model.eval()
    cer_metric = CER()
    wer_metric = WER(mode=wer_mode)

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)

            gen_kwargs = {
                "max_new_tokens": 128,
                "num_beams": 1,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": model.config.decoder_start_token_id,
            }
            gen_out = model.generate(pixel_values, **gen_kwargs)
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            sequences = sequences.detach().cpu()

            preds = processor.batch_decode(sequences, skip_special_tokens=True)
            gts = [id2gt.get(img_id, "") for img_id in batch["ids"]]

            for p, g in zip(preds, gts):
                cer_metric.update(p, g)
                wer_metric.update(p, g)

    model.train()
    return cer_metric.score(), wer_metric.score()


# ===== モデル保存 =====
def save_model_and_processor(model, processor, optimizer, epoch, out_dir, tag=None):
    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"epoch-{epoch}" if tag is None else f"{tag}-epoch-{epoch}"
    ckpt_dir = os.path.join(out_dir, tag_str)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)

    torch.save(
        {"optimizer": optimizer.state_dict(), "epoch": epoch},
        os.path.join(ckpt_dir, "optim.pt"),
        _use_new_zipfile_serialization=False,
    )
    print(f"[INFO] Saved checkpoint to {ckpt_dir}")


# ===== optimizer + scheduler 作成 =====
def create_optimizer_and_scheduler(model, train_loader, config):
    """
    encoder も学習するが、encoder の LR を小さめにする。
    """
    base_lr = float(config.train.lr)
    weight_decay = float(config.train.weight_decay)
    max_epochs = int(config.train.num_epochs)

    # encoder / decoder で LR を分ける
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": base_lr * 0.5},  # encoder は少し小さく
            {"params": decoder_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )

    num_training_steps = len(train_loader) * max_epochs
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"[INFO] Trainable params: {trainable_params} / {all_params} "
        f"({trainable_params / all_params * 100:.2f}%)"
    )
    print(
        f"[INFO] Encoder LR: {base_lr * 0.5:.2e}, "
        f"Decoder LR: {base_lr:.2e}, Warmup steps: {num_warmup_steps}"
    )

    return optimizer, scheduler


# ===== メイン =====
if __name__ == "__main__":
    config = parse_args()

    # device の決定
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    max_epochs = int(config.train.num_epochs)
    patience = int(getattr(config.train, "patience", 6))  # 追加パラメータ

    # ---------------- Processor / DataLoader ----------------
    processor = TrOCRProcessor.from_pretrained(config.model_name)

    train_loader = get_dataloader(
        config.data.train_images_dir,
        config.data.train_labels_path,
        processor,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )

    val_loader = get_dataloader(
        config.data.val_images_dir,
        config.data.val_labels_path,
        processor,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.eval.num_workers,
    )

    test_loader = get_dataloader(
        config.data.test_images_dir,
        config.data.test_labels_path,
        processor,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.eval.num_workers,
    )

    id2gt_val = build_id2gt(val_loader.dataset)
    id2gt_test = build_id2gt(test_loader.dataset)

    # ---------------- モデル読み込み ----------------
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model_name
    ).to(device)

    # 特殊トークンの設定（念のため）
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id

    # optimizer & scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, train_loader, config
    )

    # logger
    logger = HTRLogger(log_dir=config.logging.log_dir, config=config)

    # ---------------- 学習前の評価（ベースライン） ----------------
    cer_val0, wer_val0 = run_eval(
        model,
        processor,
        val_loader,
        device,
        id2gt_val,
        wer_mode=config.eval.wer_mode,
    )
    cer_test0, wer_test0 = run_eval(
        model,
        processor,
        test_loader,
        device,
        id2gt_test,
        wer_mode=config.eval.wer_mode,
    )

    print(f"[Before FT] Val  CER: {cer_val0:.4f}, WER: {wer_val0:.4f}")
    print(f"[Before FT] Test CER: {cer_test0:.4f}, WER: {wer_test0:.4f}")

    # epoch=0 としてログに入れておく
    logger.log_epoch(0, cer_val0, wer_val0, lr=optimizer.param_groups[0]["lr"])
    logger.log_test(0, cer_test0, wer_test0)

    # ---------------- トレーニングループ ----------------
    best_cer = cer_val0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
        )

        running_loss = 0.0

        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            # 勾配爆発対策
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            avg_loss = running_loss / step
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            logger.log_step(loss.item(), epoch, step)

        # ----- epoch 終了時に validation -----
        cer_val, wer_val = run_eval(
            model,
            processor,
            val_loader,
            device,
            id2gt_val,
            wer_mode=config.eval.wer_mode,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, cer_val, wer_val, lr=current_lr)

        print(f"[Epoch {epoch}] Val CER: {cer_val:.4f}, WER: {wer_val:.4f}")

        # ----- ベスト更新チェック -----
        if cer_val < best_cer:
            best_cer = cer_val
            best_epoch = epoch
            epochs_no_improve = 0

            print(
                f"[INFO] New best CER {best_cer:.4f} at epoch {epoch}. "
                "Saving model and evaluating on test set..."
            )
            save_model_and_processor(
                model,
                processor,
                optimizer,
                epoch,
                config.model.save_dir,
                tag="best",
            )

            cer_test, wer_test = run_eval(
                model,
                processor,
                test_loader,
                device,
                id2gt_test,
                wer_mode=config.eval.wer_mode,
            )
            print(f"[Epoch {epoch}] Test CER: {cer_test:.4f}, WER: {wer_test:.4f}")
            logger.log_test(epoch, cer_test, wer_test)
        else:
            epochs_no_improve += 1
            print(
                f"[INFO] No improvement for {epochs_no_improve} epoch(s). "
                f"Best CER {best_cer:.4f} at epoch {best_epoch}."
            )

            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

        # お好みで「一定間隔でスナップショットを残したい」場合
        save_interval = getattr(config.model, "save_interval", 0)
        if save_interval and (epoch % save_interval == 0):
            save_model_and_processor(
                model,
                processor,
                optimizer,
                epoch,
                config.model.save_dir,
                tag="snapshot",
            )

    logger.close()
    print(
        f"[INFO] Training finished. Best epoch: {best_epoch}, "
        f"Best Val CER: {best_cer:.4f}"
    )
