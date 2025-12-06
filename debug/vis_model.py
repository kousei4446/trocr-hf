# trocr_infer_single.py
# 実行例: python vis_model.py /path/to/image.png
# small:61596672 parameters
# base :333921792 parameters

import sys
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor



def main():

    MODEL_NAME = "microsoft/trocr-small-printed"

    # CPU / GPU 自動判定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルとプロセッサ読み込み（初回はダウンロードされます）
    # processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("☕"*100)
    print(total_params)
    print(trainable_params)
    print("☕"*100)
    print("☆"*100)
    print(f"Model '{model}' loaded.")
    print("☆"*100)
    
    #　プロセッサとは何かを確認
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    print("☕"*100)
    print(processor)
    print("☆"*100)


if __name__ == "__main__":
    main()
