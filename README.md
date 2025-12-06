# TrOCR Fine-tuning

Microsoft TrOCR モデルの Fine-tuning フレームワーク

## 概要

HuggingFace Transformers を使用した TrOCR（Transformer-based Optical Character Recognition）モデルのファインチューニングを行うためのプロジェクトです。

TrOCR は Vision Transformer (ViT) をエンコーダー、テキスト生成用の Transformer をデコーダーとして使用する End-to-End の OCR モデルです。

## 特徴

- TrOCR モデル（small/base/large）のファインチューニング
- カスタムデータセットでの学習
- TensorBoard によるロギング
- CER（Character Error Rate）/ WER（Word Error Rate）での評価
- OmegaConf による柔軟な設定管理

## プロジェクト構成

```
trocr-hf/
├── train.py              # 学習スクリプト
├── config.yaml           # 設定ファイル
├── requirements.txt      # 依存パッケージ
├── README.md
├── utils/
│   ├── dataset.py        # データセット・DataLoader
│   ├── metrics.py        # CER/WER 評価指標
│   └── logger.py         # TensorBoard ロガー
├── debug/
│   ├── test.py           # 推論テスト（フォルダ単位）
│   └── dataset.py        # データセット動作確認
└── data/
    ├── train/
    │   ├── images/       # 学習画像
    │   └── labels.txt    # 学習ラベル
    └── val/
        ├── images/       # 検証画像
        └── labels.txt    # 検証ラベル
```

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd trocr-hf

# 依存パッケージのインストール
pip install -r requirements.txt

# NLTK データのダウンロード（初回のみ）
python -c "import nltk; nltk.download('punkt')"
```

## データ形式

### ラベルファイル（labels.txt）

```
image_id1 テキスト内容1
image_id2 テキスト内容2
image_id3 テキスト内容3
```

- 各行: `<画像ID> <テキスト>`（スペース区切り）
- 画像ID は拡張子なし（例: `img001`）
- 対応する画像は `images/<画像ID>.png` に配置

### ディレクトリ構成例

```
data/train/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── labels.txt
```

## 使用方法

### 学習

```bash
python train.py config.yaml
```

CLI からパラメータを上書き可能:

```bash
python train.py config.yaml train.lr=5e-5 train.batch_size=16
```

### 推論テスト

```bash
# フォルダ内の全画像をテスト
python debug/test.py data/val/images --model microsoft/trocr-small-handwritten

# 詳細出力
python debug/test.py data/val/images --model microsoft/trocr-small-handwritten -v
```

### TensorBoard でログ確認

```bash
tensorboard --logdir=logs
```

## 設定ファイル（config.yaml）

```yaml
model_name: 'microsoft/trocr-small-handwritten'
device: 'cuda'

data:
  train_images_dir: 'data/train/images'
  train_labels_path: 'data/train/labels.txt'
  val_images_dir: 'data/val/images'
  val_labels_path: 'data/val/labels.txt'

train:
  num_epochs: 30
  batch_size: 8
  num_workers: 4
  lr: 1e-4
  weight_decay: 0.00005

eval:
  batch_size: 8
  num_workers: 4
  wer_mode: 'tokenizer'  # 'tokenizer' or 'space'

model:
  save_dir: './models/saved_models/TOCR-small'
  save_interval: 10

logging:
  log_dir: './logs/TOCR-small'
```

## 利用可能なモデル

| モデル名 | パラメータ数 | 用途 |
|----------|-------------|------|
| `microsoft/trocr-small-handwritten` | 62M | 手書き文字（軽量） |
| `microsoft/trocr-base-handwritten` | 334M | 手書き文字（標準） |
| `microsoft/trocr-large-handwritten` | 558M | 手書き文字（高精度） |
| `microsoft/trocr-small-printed` | 62M | 印刷文字（軽量） |
| `microsoft/trocr-base-printed` | 334M | 印刷文字（標準） |

## 評価指標

- **CER（Character Error Rate）**: 文字単位の編集距離
- **WER（Word Error Rate）**: 単語単位の編集距離
  - `tokenizer` モード: NLTK でトークナイズ
  - `space` モード: スペースで分割

## 動作環境

- Python 3.10+
- CUDA 11.8+（GPU使用時）
- PyTorch 2.0+

## 注意事項

- GPU メモリが不足する場合は `batch_size` を小さくしてください
- Windows で `num_workers > 0` を使用する場合、データセット内に lambda 関数があるとエラーになります
- 日本語テキストの場合、トークナイザーの調整が必要な場合があります

## ライセンス

MIT License
