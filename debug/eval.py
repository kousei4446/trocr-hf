"""
TrOCR デバッグ用スクリプト（逐次確認）
使い方:
"""

import sys
import os

# プロジェクトルートを import パスに追加（utils などを読み込むため）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import argparse
import math
from utils.metrics import CER, WER

from omegaconf import OmegaConf

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def safe_load_image(path):
    """
    画像を安全に読み込み、RGB に変換して返す
    """
    img = Image.open(path).convert("RGB")
    return img


def find_ground_truth(labels_txt_path, image_id):
    """
    labels.txt から指定 image_id に対応する正解テキストを取得する
    フォーマット: image_id テキスト内容（例: "c04-110-00 Some text here"）
    先頭が "image_id + 半角スペース" で始まる行を探索し、右側のテキストを返す
    見つからなければ None
    """
    with open(labels_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n\r")
            if s.startswith(image_id + " "):
                # image_id の後ろのテキスト全体を返す
                return s.split(None, 1)[1] if len(s.split(None, 1)) > 1 else ""
    return None


def debug_inference_folder(images_dir: str, model_name: str , device=None):
    """
    指定フォルダ内の全画像を推論し、集計指標（CER/WER）を算出する
    - images_dir: 画像フォルダへのパス
    - model_name: 使用する HuggingFace の TrOCR モデル名
    - device: 使用デバイス（未指定なら CUDA があれば cuda、なければ cpu）
    """
    # デバイス自動選択
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # プロセッサ（前処理）とモデルの読み込み
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    # 重要な設定値の表示（デコーダの開始トークン / パディングトークン）
    cfg = model.config
    print(f"decoder_start_token_id: {getattr(cfg, 'decoder_start_token_id', None)}, pad_token_id: {getattr(cfg, 'pad_token_id', None)}")

    # images_dir の親ディレクトリにある labels.txt を参照
    labels_txt_path = os.path.join(os.path.dirname(images_dir), "labels.txt")
    if not os.path.exists(labels_txt_path):
        raise FileNotFoundError(f"labels.txt が見つかりません: {labels_txt_path}")

    # フォルダ内の画像ファイル一覧を取得（拡張子でフィルタ）
    # フォルダ内 pngのみ対応
    image_extensions = {'.png'}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f"{images_dir} に画像ファイルが存在しません")

    print(f"{images_dir} で {len(image_files)} 枚の画像を検出")
    print("=" * 80)

    # 集計用メトリクス（全体 CER/WER）
    cer_metric = CER()
    wer_metric = WER(mode='tokenizer')

    # 生成時のパラメータ（None のものは除外）
    gen_kwargs = {
        "max_new_tokens": 64,                  # 最大生成トークン数
        "num_beams": 1,                        # ビームサーチ幅
        "pad_token_id": getattr(cfg, "pad_token_id", None),
        "decoder_start_token_id": getattr(cfg, "decoder_start_token_id", None),
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    results = []  # 各サンプルの結果を保存

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        image_id = os.path.splitext(image_file)[0]

        # 画像読み込みと前処理（pixel_values を取得）
        img = safe_load_image(image_path)
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

        # 正解テキストの取得
        gt_text = find_ground_truth(labels_txt_path, image_id)
        if gt_text is None:
            print(f"[{idx+1}/{len(image_files)}] {image_id}: 正解テキストが見つからないためスキップ")
            continue

        # 推論（生成）
        with torch.no_grad():
            # generate は設定によって GenerateOutput または Tensor を返すことがある
            gen_out = model.generate(pixel_values, **gen_kwargs)

        # 戻り値の型に応じて sequences を取り出す
        if hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
        else:
            sequences = gen_out  # テンソルが直接返るケース

        # CPU に移してテキストへデコード
        sequences_cpu = sequences.detach().cpu()
        decoded = processor.batch_decode(sequences_cpu, skip_special_tokens=True)[0]

        # 集計メトリクス更新
        cer_metric.update(decoded, gt_text)
        wer_metric.update(decoded, gt_text)

        # サンプル単位の CER/WER を計算（個別保存用）
        sample_cer = CER()
        sample_wer = WER(config.eval.wer_mode)
        sample_cer.update(decoded, gt_text)
        sample_wer.update(decoded, gt_text)

        results.append({
            'image_id': image_id,
            'gt': gt_text,
            'pred': decoded,
            'cer': sample_cer.score(),
            'wer': sample_wer.score(),
        })
        print(f"\r[{idx+1}/{len(image_files)}] Processing...", end="", flush=True)

    print()  # 行末の改行

    # サマリ表示（全体 CER/WER）
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Overall CER: {cer_metric.score():.4f}")
    print(f"Overall WER: {wer_metric.score():.4f}")

    # ワースト上位5件（CERの高い順）を表示
    if results:
        print("\n--- Worst 5 samples by CER ---")
        sorted_by_cer = sorted(results, key=lambda x: x['cer'], reverse=True)[:5]
        for r in sorted_by_cer:
            print(f"{r['image_id']}: CER={r['cer']:.4f}")
            print(f"  GT:   {r['gt']}")
            print(f"  Pred: {r['pred']}")

    # 結果リスト、全体 CER、全体 WER を返す
    return results, cer_metric.score(), wer_metric.score()


if __name__ == "__main__":      
    # 引数パーサ
    p = argparse.ArgumentParser()
    config = parse_args()
    
    # 例外を捕捉して見やすく表示
    try:
        debug_inference_folder(config.data.test_images_dir, config.model_name,config.device)
    except Exception as e:
        print("推論デバッグ中にエラーが発生しました:", e)
        raise
