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
├── train.py                  # 学習スクリプト
├── train_pro.py              # 追加の学習スクリプト
├── config.yaml               # 設定ファイル
├── requirements.txt          # 依存パッケージ
├── data/                     # 前処理後データ配置先（train/val/test 配下に images, labels.txt）
├── data_raw/
│   ├── prepare_iam.py        # IAM 前処理スクリプト（行画像＋labels.txt生成）
│   └── IAM/                  # ダウンロードした元データ配置 (forms/, xml/, splits/)
├── utils/
│   ├── dataset.py            # データセット・DataLoader
│   ├── metrics.py            # CER/WER 評価指標
│   └── logger.py             # TensorBoard ロガー
├── debug/
│   ├── test.py               # 推論テスト（フォルダ単位）
│   └── dataset.py            # データセット動作確認
├── logs/                     # 学習ログ出力先
└── models/                   # モデル保存先
```

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd trocr-hf

# 依存パッケージのインストール
pip install -r requirements.txt
```

## データ準備

### IAM データを使う場合

1. IAM 公式サイトで登録し、以下を取得して解凍します（配布物名は公式に準拠）  
   - フォーム画像: `data/formsA-D.tgz`, `data/formsE-H.tgz`, `data/formsI-Z.tgz` → 共通の `forms/` にまとめて展開（配布元: https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database ）  
   - XML Ground Truth: `data/xml.tgz` → `xml/` に展開  
   - 行分割リスト: `train.uttlist`, `validation.uttlist`, `test.uttlist`（本リポジトリでは `data_raw/IAM/splits/` に配置済み）

2. ディレクトリ例（リポジトリ配下に置く場合）  
   ```
   data_raw/IAM/
     ├── forms/        # 解凍したフォーム画像 (png)
     ├── xml/          # 解凍した XML
     └── splits/       # train/val/test の uttlist
   ```

3. 行画像＋`labels.txt` を生成  
   ```bash
   python data_raw/prepare_iam.py \
     ./data_raw/IAM/forms/ \
     ./data_raw/IAM/xml/ \
     ./data_raw/IAM/splits/ \
     ./data/IAM
   ```
   - 出力: `./data/IAM/{train,val,test}/images/*.png` と `labels.txt`
   - `config.yaml` の `data.*` をこの出力パスに合わせて設定

4. 注意点  
   - 公式配布の行画像は単語単位の背景マスクや欠損行があるため、本スクリプトはフォーム画像＋XMLから行を切り出します。より生のノイズを含む行画像で学習したい場合に有効です。

### 出力データ形式

- 配置: `data/{train,val,test}/images/` に画像、同階層に `labels.txt`
- `labels.txt` 各行: `<画像ID> <テキスト>`（画像IDは拡張子なし、画像は `images/<画像ID>.png`）
## 使用方法

### 学習

```bash
python train.py config.yaml
```


### TensorBoard でログ確認

```bash
tensorboard --logdir=logs
```


## 利用可能なモデル

| モデル名 | パラメータ数 | 用途 |
|----------|-------------|------|
| `microsoft/trocr-small-handwritten` | 62M | 手書き文字（軽量） |
| `microsoft/trocr-base-handwritten` | 334M | 手書き文字（標準） |
| `microsoft/trocr-large-handwritten` | 558M | 手書き文字（高精度） |

## 評価指標

- **CER（Character Error Rate）**: 文字単位の編集距離
- **WER（Word Error Rate）**: 単語単位の編集距離
  - `tokenizer` モード: NLTK でトークナイズ
  - `space` モード: スペースで分割

## 動作環境

- Python 3.10+
- CUDA 11.8+（GPU使用時）
- PyTorch 2.0+
