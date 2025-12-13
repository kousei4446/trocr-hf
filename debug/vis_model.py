# debug/vis_model.py

from models.trocr_small import TrOCR_SMALL
import sys
import os
from omegaconf import OmegaConf
import torch
from transformers import VisionEncoderDecoderConfig


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


def main():
    # ① 事前に保存しておいた state_dict のパス
    WEIGHT_PATH = os.path.join("saved_models","trocr-small-handwritten", "trocr-small-handwritten-state.pt")

    # ② 学習用 config.yaml（これは device などにだけ使う）
    train_conf = parse_args()
    device = train_conf.device if torch.cuda.is_available() else "cpu"

    # ③ Hugging Face のモデル config（こっちを TrOCR_SMALL に渡す）
    HF_MODEL_NAME = "microsoft/trocr-small-stage1"
    hf_config = VisionEncoderDecoderConfig.from_pretrained(HF_MODEL_NAME)

    # ④ 自作モデルを HF config で初期化
    my_model = TrOCR_SMALL(hf_config).to(device)
    
    
    
    print("☆" * 100)    
    print(my_model)
    # print(hf_config)
    print("☆" * 100)

    # ⑤ 保存済み state_dict をロード
    print(f"Loading state_dict from: {WEIGHT_PATH}")
    state_dict = torch.load(WEIGHT_PATH, map_location=device)

    try:
        my_model.load_state_dict(state_dict, strict=True)
        print("✅ strict=True で load_state_dict 成功")
    except RuntimeError as e:
        print("❌ strict=True でエラー発生:")
        print(e)

        print("\n🔍 strict=False でもう一度ロードして差分確認:")
        incompatible = my_model.load_state_dict(state_dict, strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys

        print("\nMissing keys（自作モデルにあるのに state_dict に無いキー）:")
        for k in missing:
            print("  ", k)

        print("\nUnexpected keys（state_dict にあるのに 自作モデルに無いキー）:")
        for k in unexpected:
            print("  ", k)


if __name__ == "__main__":
    main()
