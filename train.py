# 警告を抑制
import warnings
import logging
import os

# 環境変数で Transformers のログを抑制（import前に設定）
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Transformers の警告を抑制
warnings.filterwarnings("ignore", message=".*as_target_tokenizer.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*Some weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*Config of the.*is overwritten.*")

# HuggingFace のロギングレベルを設定
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)



from omegaconf import OmegaConf

import sys

from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from utils.dataset import get_dataloader

from utils.metrics import CER, WER
from utils.logger import HTRLogger

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def evaluate(model, processor, loader, device, id2gt, wer_mode):
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

            # 事前に作った id->gt 辞書を使う（高速）
            gts = [id2gt.get(img_id, "") for img_id in batch["ids"]]

            for p, g in zip(preds, gts):
                cer_metric.update(p, g)
                wer_metric.update(p, g)
    model.train()
    return cer_metric.score(), wer_metric.score()         


def test(model, processor, test_loader, device, wer_mode):
    model.eval()
    cer_metric = CER()
    wer_metric = WER(mode=wer_mode)
    with torch.no_grad():
        for batch in test_loader:
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

            gts = [batch["labels"][i][batch["labels"][i] != -100].tolist() for i in range(batch["labels"].size(0))]
            gts = processor.tokenizer.batch_decode(gts, skip_special_tokens=True)

            for p, g in zip(preds, gts):
                cer_metric.update(p, g)
                wer_metric.update(p, g)
    model.train()
    return cer_metric.score(), wer_metric.score()

    

def save_model_and_processor(model, processor, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # HuggingFace 形式で保存（復元が楽）
    ckpt_dir = os.path.join(out_dir, f"epoch-{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    # optimizer state 保存（再開用）
    
    torch.save(
      {"optimizer": optimizer.state_dict(), "epoch": epoch},
      os.path.join(ckpt_dir, "optim.pt"),
      _use_new_zipfile_serialization=False
    )
    print(f"Saved checkpoint to {ckpt_dir}")
 

if __name__ == "__main__":
    config = parse_args()
    max_epochs = config.train.num_epochs
    device = config.device    
    
    processor = TrOCRProcessor.from_pretrained(config.model_name)
    train_loader = get_dataloader(config.data.train_images_dir, config.data.train_labels_path, processor, batch_size=config.train.batch_size, shuffle=True,num_workers=config.train.num_workers )
    val_loader = get_dataloader(config.data.val_images_dir, config.data.val_labels_path, processor, batch_size=config.eval.batch_size, shuffle=False, num_workers=config.eval.num_workers )
    test_loader = get_dataloader(config.data.test_images_dir, config.data.test_labels_path, processor, batch_size=config.eval.batch_size, shuffle=False, num_workers=config.eval.num_workers)
    
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name).to(device)
    
      # デコーダの特殊トークンを設定
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr,weight_decay=config.train.weight_decay)
    id2gt = {iid: txt for (pth, txt, iid) in val_loader.dataset.samples}
    logger = HTRLogger(log_dir=config.logging.log_dir, config=config)
    
    cer_test, wer_test = test(model, processor, test_loader, device, wer_mode=config.eval.wer_mode)
    print(f"Initial Test CER: {cer_test:.4f}, WER: {wer_test:.4f}")
    avg_loss = logger.log_test(1, cer_test, wer_test)
    
    for epoch in range(1,max_epochs+1):
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        loss_val = 0.0
        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
                        
            optimizer.step()
            optimizer.zero_grad()
            loss_val = loss_val + loss.item()
            pbar.set_postfix({"loss": f"{(loss_val / step):.4f}"})
            logger.log_step(loss.item(), epoch, step)
            
        cer, wer = evaluate(model, processor, val_loader, device, id2gt, wer_mode=config.eval.wer_mode)
        print(f"Epoch {epoch} : {cer:.4f}, WER: {wer:.4f}")
        current_lr = optimizer.param_groups[0]["lr"]  
        avg_loss = logger.log_epoch(epoch, cer, wer, lr=current_lr)
        
        if epoch == 20 or epoch == 100:
            save_model_and_processor(model, processor, optimizer, epoch, config.model.save_dir)
            
        if epoch % 10 == 0:
            cer_test, wer_test = test(model, processor, test_loader, device, wer_mode=config.eval.wer_mode)
            print(f"Test CER: {cer_test:.4f}, WER: {wer_test:.4f}")
            logger.log_test(epoch, cer_test, wer_test)
            
    logger.close()