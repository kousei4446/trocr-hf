import logging
import os
import sys
import warnings

from PIL import Image
from tqdm import tqdm

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from omegaconf import OmegaConf
import torch.nn.functional as F

from utils.dataset import get_dataloader
from utils.metrics import CER, normalize_36_charset
from utils.logger import HTRLogger

from utils.transforms import build_train_transform

# Transformers/tokenizers の警告を抑制
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
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "epoch": epoch},
        os.path.join(ckpt_dir, "optim.pt"),
        _use_new_zipfile_serialization=False,
    )
    print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    config = parse_args()

    max_epochs = config.train.num_epochs
    device = config.device if torch.cuda.is_available() else "cpu"

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

    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    
    # # ===== 全パラメータ凍結 =====
    # for p in model.parameters():
    #     p.requires_grad = False
    # last_encoder_layer = model.encoder.encoder.layer[-1]
    # for p in last_encoder_layer.parameters():
    #     p.requires_grad = True
    # decoder_core = model.decoder.model.decoder  # TrOCRDecoder                                            
    # for layer in decoder_core.layers[-2:]:      # 例: 後ろ2層                                             
    #     for p in layer.parameters():                                                                      
    #         p.requires_grad = True   
            
    model.config.pad_token_id = 1
    model.config.eos_token_id = 2
    model.config.decoder_start_token_id = 2
    model = model.to(device)

    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    # 評価用の id->gt 辞書
    id2gt_val = {iid: txt for (_p, txt, iid) in val_loader.dataset.samples}
    id2gt_test = {iid: txt for (_p, txt, iid) in test_loader.dataset.samples}
    logger = HTRLogger(log_dir=config.logging.log_dir, config=config)
    # cer_test = evaluate(
    #     model,
    #     processor,
    #     test_loader,
    #     device,
    #     id2gt_test,
    #     desc=f"Test epoch {0}",
    # )
    # print(f"Test CER: {cer_test:.4f}")
    # logger.log_test(0, cer_test)
        
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
            
            logits = outputs.logits
            vocab_size = logits.size(-1)
            
            loss = F.cross_entropy(
                # 予測された“各文字位置 × 語彙数”の生のスコア（logits）
                logits.view(-1, vocab_size),
                # 正解ラベルを平坦化したもの
                labels.view(-1),   
                ignore_index=-100,
            )
            
            # loss = outputs.loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_val += loss.item()
            pbar.set_postfix({"loss": f"{(loss_val / step):.4f}"})
            logger.log_step(loss.item(), epoch, step)
            
        if epoch % config.train.save_interval == 0:
            save_model_and_processor(model, processor, optimizer, epoch, config.model.save_dir)

        if epoch % config.train.eval_interval == 0 :
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
