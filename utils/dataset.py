# -*- coding: utf-8 -*-
"""
dataset.py - TrOCR学習用データセットモジュール

画像とテキストラベルのペアを読み込み、DataLoaderを生成する。
"""
from typing import List, Optional, Tuple, Dict, Any
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torch

import functools

def _identity(x):
    """恒等関数（何もせずそのまま返す）"""
    return x

class IAMDataset(Dataset):
    """
    OCR用データセットクラス

    画像ディレクトリとラベルファイルから、画像と対応するテキストラベルを読み込む。

    ラベルファイルのフォーマット:
        各行: <image_id> <text>
        例: "img001 Hello World"

    画像ファイルの配置:
        <images_dir>/<image_id><image_ext>
        例: images/img001.png

    Attributes:
        samples: (画像パス, テキスト, 画像ID) のタプルのリスト
    """

    def __init__(self, images_dir: str, labels_txt: str, normalize_fn=None, image_ext: str = ".png"):
        """
        Args:
            images_dir: 画像ファイルが格納されているディレクトリパス
            labels_txt: ラベルファイル（各行: image_id text）のパス
            normalize_fn: テキストに適用する正規化関数（Noneの場合は何もしない）
            image_ext: 画像ファイルの拡張子（デフォルト: .png）
        """
        self.images_dir = images_dir
        # 正規化関数が指定されていなければ恒等関数を使用
        self.normalize_fn = normalize_fn if normalize_fn is not None else _identity
        self.image_ext = image_ext
        self.samples = []  # (image_path, text, image_id) のリスト

        # ラベルファイルを読み込み、サンプルリストを構築
        with open(labels_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                # 空行をスキップ
                if not line:
                    continue
                # 最初のスペースで分割: [image_id, text]
                parts = line.split(None, 1)
                if len(parts) == 0:
                    continue
                image_id = parts[0]
                # テキストがない場合は空文字列
                text = parts[1] if len(parts) > 1 else ""
                img_path = os.path.join(images_dir, image_id + image_ext)
                # 画像ファイルが存在する場合のみサンプルに追加
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.normalize_fn(text), image_id))
                # 存在しない画像はスキップ（警告を出す場合はここに追加）

    def __len__(self) -> int:
        """データセットのサンプル数を返す"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        指定インデックスのサンプルを取得

        Returns:
            dict: {
                "image": PIL.Image (RGB),
                "text": 正解テキスト,
                "id": 画像ID,
                "image_path": 画像ファイルパス
            }
        """
        img_path, text, image_id = self.samples[idx]
        image = load_rgb(img_path)
        return {"image": image, "text": text, "id": image_id, "image_path": img_path}


def load_rgb(path: str) -> Image.Image:
    """
    画像を読み込み、RGB形式で返す

    処理内容:
        1. 画像ファイルを開く
        2. EXIF情報に基づいて回転を補正（スマホ撮影画像対応）
        3. RGB形式に変換（グレースケールやRGBA画像に対応）

    Args:
        path: 画像ファイルのパス

    Returns:
        PIL.Image: RGB形式の画像
    """
    img = Image.open(path)
    # EXIF情報による回転補正（スマホで撮影した画像の向きを正しくする）
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # RGB形式に変換（グレースケールやRGBAをRGBに統一）
    img = img.convert("RGB")
    return img


def collate_fn(batch: List[Dict[str, Any]], processor, max_target_length: int = 128) -> Dict[str, Any]:
    """
    バッチデータをモデル入力形式に変換するcollate関数

    処理内容:
        1. 画像をTrOCRProcessor経由でpixel_valuesに変換
        2. テキストをトークナイズしてlabelsに変換
        3. パディングトークンを-100に置換（損失計算から除外するため）

    Args:
        batch: DataLoaderから渡されるサンプルのリスト
               各要素: {"image": PIL.Image, "text": str, "id": str}
        processor: TrOCRProcessor（画像処理とトークナイザを含む）
        max_target_length: テキストの最大トークン長

    Returns:
        dict: {
            "pixel_values": 画像テンソル (B, C, H, W),
            "labels": ラベルテンソル (B, L)、パディング部分は-100,
            "ids": 画像IDのリスト
        }
    """
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    ids = [x["id"] for x in batch]

    # エンコーダ入力: 画像をリサイズ・正規化してテンソルに変換
    encodings = processor(images=images, return_tensors="pt")
    pixel_values = encodings.pixel_values  # (B, C, H, W)

    # デコーダ入力: テキストをトークナイズ（text_target引数を使用）
    tokenizer = processor.tokenizer
    labels = tokenizer(
        text_target=texts,
        padding="longest",          # バッチ内で最長に合わせてパディング
        truncation=True,            # 最大長を超える場合は切り詰め
        max_length=max_target_length,
        return_tensors="pt"
    ).input_ids

    # パディングトークンを-100に置換
    # CrossEntropyLossはignore_index=-100のため、パディング部分の損失が無視される
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels[labels == pad_id] = -100

    batch_out = {
        "pixel_values": pixel_values,     # 画像テンソル (B, C, H, W)
        "labels": labels,                 # ラベルテンソル (B, L)、パディングは-100
        "ids": ids                        # 画像IDリスト（評価時に使用）
    }
    return batch_out


def get_dataloader(images_dir: str, labels_txt: str, processor, batch_size: int = 8, shuffle: bool = True,
                   num_workers: int = 4, max_target_length: int = 128, image_ext: str = ".png") -> DataLoader:
    """
    学習/評価用DataLoaderを生成

    Args:
        images_dir: 画像ディレクトリのパス
        labels_txt: ラベルファイルのパス
        processor: TrOCRProcessor
        batch_size: バッチサイズ
        shuffle: データをシャッフルするか（学習時True、評価時False）
        num_workers: データ読み込みの並列ワーカー数
        max_target_length: テキストの最大トークン長

    Returns:
        DataLoader: 設定済みのDataLoader
    """
    ds = IAMDataset(images_dir, labels_txt, normalize_fn=None, image_ext=image_ext)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_fn, processor=processor, max_target_length=max_target_length)
    )
