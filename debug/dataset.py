"""
実行コマンド：python -m debug.dataset config.yaml
"""
from omegaconf import OmegaConf

import sys

from transformers import  TrOCRProcessor

from utils.dataset import get_dataloader

from tqdm import tqdm



import os
import numpy as np
from PIL import Image





def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf




if __name__ == "__main__":
    config = parse_args()
    os.makedirs(config.debug.save_dir, exist_ok=True)
    
    max_epochs = config.train.num_epochs
    device = config.device    
    
    processor = TrOCRProcessor.from_pretrained(config.model_name)
    train_loader = get_dataloader(config.data.dummy_image_dir, config.data.dummy_label_path, processor, batch_size=config.train.batch_size, shuffle=True,num_workers=config.train.num_workers )
    print("データローダーの内容を確認")
    for batch in train_loader:
        print("☆"*20)
        print(batch)
        break
    print("☆"*100)
    for epoch in range(1,2):
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            print("----")
            print("バッチ内のテンソル形状を確認")
            print(batch)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            ids = batch.get("ids",None)
            
            pv_batch = pixel_values.detach().cpu() # (B ,C,H,W)
            B = pv_batch.size(0)
            saved_count = 0
            for i in range(B):
                pv = pv_batch[i]
                img_mp = pv.permute(1,2,0).numpy()  # (H,W,C)
                img_pil = Image.fromarray((img_mp * 255).astype(np.uint8))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_mp = img_mp * std + mean  # 逆正規化
                img_mp = np.clip(img_mp, 0, 1)  # 0-1にクリップ
                img_pil = Image.fromarray((img_mp * 255).astype(np.uint8))


                # ファイル名作成
                sample_id = ids[i] if ids is not None and i < len(ids) else f"b{step}i{i}"
                fname = f"ep{epoch:02d}_st{step:04d}_{i:03d}_{sample_id}.png"
                outpath = os.path.join(config.debug.save_dir, fname)
                img_pil.save(outpath)
                saved_count += 1

            # ログ更新
            pbar.set_postfix({"saved_imgs": saved_count})
            
            
            
            
            print("pixel_values :", pixel_values)
            print("pixel_values shape:", pixel_values.shape)
            
            print("labels :", labels.shape)
            print("labels shape:", labels)
            
            break
        break